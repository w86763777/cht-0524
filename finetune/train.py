# Built-in packages
import os
import warnings

# 3rd party packages
import torch
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer


if __name__ == '__main__':
    logdir = './checkpoints'
    device = torch.device('cuda:0')

    # Disable FutureWarnings from `huggingface_hub`.
    warnings.simplefilter(action='ignore', category=FutureWarning)

    def formatting_train(sample):
        # sample['document']: news content
        # sample['summary']: news summary
        text = ""
        return {'text': text}

    dataset = Dataset.from_csv('./dataset.csv')
    dataset = dataset.map(formatting_train, num_proc=2)
    datasets = dataset.train_test_split(test_size=0.2, seed=0)   # set seed for reproducibility

    model_name = "meta-llama/Llama-2-7b-hf"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map={"": 0},
    )

    # doc: https://huggingface.co/docs/peft/package_reference/peft_model#peft.prepare_model_for_kbit_training
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": True}   # silence warning
    )
    # doc: https://huggingface.co/docs/peft/package_reference/lora#peft.LoraConfig
    lora_config = LoraConfig(
        r=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"],
        lora_alpha=16,
        lora_dropout=0.05,
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.config.use_cache = False                              # silence the warnings. Please re-enable for inference!

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token                   # The pretrained tokenizer for llama2 lacks of padding token
    tokenizer.add_eos_token = True                              # Add eos token to the end of text is important for training

    training_arguments = TrainingArguments(
        output_dir=logdir,
        max_grad_norm=0.3,
        warmup_ratio=0.3,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,

        optim="paged_adamw_32bit",
        learning_rate=2e-4,
        lr_scheduler_type="linear",

        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),

        logging_steps=100,
        save_total_limit=3,
        save_strategy="steps",
        save_steps=500,
        evaluation_strategy="steps",
        eval_steps=500,
        load_best_model_at_end=True,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=datasets['train'],
        eval_dataset=datasets['test'],
        tokenizer=tokenizer,
        args=training_arguments,

        # doc: https://huggingface.co/docs/trl/sft_trainer#trl.SFTTrainer.packing
        packing=True,
        dataset_text_field="text",
        max_seq_length=512,
    )

    trainer.train()
    trainer.save_model(os.path.join(logdir, 'best'))
