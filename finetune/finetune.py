import warnings

import re
import os
import torch
import phonemizer
import wandb

from fire import Fire
from omegaconf import OmegaConf
from functools import partial
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, default_data_collator
from loguru import logger as LOGGER
from datasets import load_dataset


warnings.filterwarnings("ignore")


ACRONYM = re.compile(r"(?:[a-zA-Z]\.){2,}")
ACRONYM_NO_PERIOD = re.compile(r"(?:[A-Z]){2,}")


def data_filter(sample):
    text = sample["text"]

    if len(text) == 0:
        return False

    if re.search(r'\d', text):
        return False

    if re.search(ACRONYM, text) or re.search(ACRONYM_NO_PERIOD, text):
        return False

    if text[-1] not in ".,?!":
        return False

    if '£' in text or '$' in text:
        return False

    return True


def preprocess_sample(sample, tokenizer, max_len, g2p):

    # get special tokens
    speech_gen_start = tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_START|>')
    ignore_index = -100  # this is from LLaMA

    # unpack sample
    vq_codes = sample["codes"]
    text = sample["text"]

    # phonemize
    phones = g2p.phonemize([text])


    # This is the dictionary to return on failure
    invalid_sample_output = {"input_ids": [], "labels": [], "valid": False}

    # SAFE CHECK
    if not phones or not phones[0]:
        LOGGER.warning(f"⚠️ Empty phonemization output for sample: {sample['__key__']} text={text}")
        return invalid_sample_output

    phones = phones[0].split()
    phones = ' '.join(phones)



    # without conditioning tokens (let's see)
    # ref_conditioned = f"<|{ref_gender}|><|{ref_dialect}|><|{ref_tone}|> {ref_phones}"
    # ref_conditioned = f"{phones}"
    # combined_text = f"{ref_conditioned}"


    # Create codes strings
    codes_str = "".join([f"<|speech_{i}|>" for i in vq_codes])

    # get chat format
    chat = f"""user: Convert the text to speech:<|TEXT_PROMPT_START|>{phones}<|TEXT_PROMPT_END|>\nassistant:<|SPEECH_GENERATION_START|>{codes_str}<|SPEECH_GENERATION_END|>"""



    ids = tokenizer.encode(chat)
    # ids = encoding['input_ids']

    # pad to make seq len
    # If sequence was too long and got truncated, mark as invalid
    # (We want complete samples, not truncated ones)
    if len(ids) >= max_len:
        LOGGER.warning(f"⚠️ Sample exceeds max_len ({len(ids)} >= {max_len}), skipping")
        return invalid_sample_output
    if len(ids) < max_len:
        ids = ids + [tokenizer.pad_token_id] * (max_len - len(ids))
    else:
        ids = ids[:max_len]

    # convert to tensor
    input_ids = torch.tensor(ids, dtype=torch.long)

    # Create labels (only train on speech generation part)


    #Create a label mask ignoring all tokens, find where speech begins, 
    # then we will later apply loss only after that point
    labels = torch.full_like(input_ids, ignore_index)
    ''' 
        Initialize labels so that all positions are ignored by loss by default.

        This is common in autoregressive training where only some tokens should contribute to loss
        (e.g., audio tokens, not text tokens).
    '''
    speech_gen_start_idx = (input_ids == speech_gen_start).nonzero(as_tuple=True)[0]
    ''' 
        (input_ids == speech_gen_start) → boolean mask of where special token appears
        Example:
        input: [1, 1556, 32000, 45001, 45002]
        speech_gen_start = 32000
        mask: [False, False, True, False, False]

        .nonzero(as_tuple=True) → get indices where mask is True
        Result: (tensor([2]), tensor([0])) depending on shape (batch, seq)

        [0] → first dimension (position index in sequence)
    '''
    if len(speech_gen_start_idx) > 0:
        speech_gen_start_idx = speech_gen_start_idx[0]

        ''' 
        Everything BEFORE <AUDIO_START> is ignored for loss
        Everything AFTER <AUDIO_START> is supervised.
        '''
        labels[speech_gen_start_idx:] = input_ids[speech_gen_start_idx:]

    # create attention mask for non real tokens like padded tokens
    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    # return in hf format
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


def main(config_fpath: str):

    # load config
    print(f"Loading config from {config_fpath}")
    config = OmegaConf.load(config_fpath)
    checkpoints_dir = os.path.join(config.save_root, config.run_name)
    LOGGER.info(f"Logging to: {checkpoints_dir}")

    restore_from = config.restore_from

    print(f"Loading checkpoint from {restore_from}")
    tokenizer = AutoTokenizer.from_pretrained(restore_from)
    model = AutoModelForCausalLM.from_pretrained(restore_from, 
                                                 torch_dtype="auto",
                                                 low_cpu_mem_usage = True)
    print(f"Model loaded: {model.num_parameters():,} Parameters")

    try:
        g2p = phonemizer.backend.EspeakBackend(
        language='hi',
        preserve_punctuation=True,
        with_stress=True,
        words_mismatch="ignore",
        language_switch="remove-flags"
        )
        # Test phonemizer 
        test_result = g2p.phonemize(["नमस्ते दुनिया"])
        print(f"✓ Phonemizer initialized successfully!")
        print(f"  Test: 'नमस्ते दुनिया' → {test_result}")
    except Exception as e:
        print(f"❌ Error initializing phonemizer: {e}")
        print("Make sure espeak-ng is installed with Hindi support:")
        print("  Ubuntu/Debian: sudo apt-get install espeak-ng")
        print("  Windows: Download from https://github.com/espeak-ng/espeak-ng/releases")
        raise


    partial_preprocess = partial(
        preprocess_sample,
        tokenizer=tokenizer,
        max_len=config.max_seq_len,
        g2p=g2p,
    )

    emilia_dataset = load_dataset(
        "speechriv/neucodec-data-hindi",
        split="train",
    )
    emilia_dataset = emilia_dataset.filter(data_filter).map(partial_preprocess, 
                                                            remove_columns=["text", "codes"], 
                                                            num_proc = 40,
                                                            desc = "Preprocessing")
    wandb.init(
        project="NeuTTS",
        name="tts-common-hindi-4",
        config={
            "model": config.restore_from,
            "train_dataset": "speechriv/neucodec-data-hindi",
            # "val_dataset": "1000_samples",
            "train_size": len(emilia_dataset),
            # "val_size": len(dataset['test']),
            "learning_rate": config.lr,
            "batch_size": config.per_device_train_batch_size,
            "epochs": config.num_train_epochs,
        }
    )
    training_args = TrainingArguments(
        output_dir=checkpoints_dir,
        do_train=True,
        learning_rate=config.lr,
        max_steps=config.max_steps,
        bf16=True,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_ratio=config.warmup_ratio,
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        save_strategy="steps",
        ignore_data_skip=True,
        dataloader_drop_last=True,
        remove_unused_columns=False,
        torch_compile=True,
        num_train_epochs = config.num_train_epochs,
        report_to=["tensorboard", "wandb"], 
        # dataloader_num_workers=64,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=emilia_dataset,
        data_collator=default_data_collator,
    )
    trainer.train()
    trainer.save_model(checkpoints_dir)
    tokenizer.save_pretrained(checkpoints_dir)


if __name__ == "__main__":
    Fire(main)