import argparse
import warnings

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model-path', type=str,
                        help='Path to the checkpoint directory for QLoRA weights.')
    parser.add_argument('--base-model', type=str, default="meta-llama/Llama-2-7b-hf",
                        help='Model name or path to the base model.')
    parser.add_argument('--out', type=str,
                        help='Path to the output directory for the merged model.')
    parser.add_argument('--device', type=str, default="cpu",
                        help='Device to use for loading the model.')
    args = parser.parse_args()

    # Disable FutureWarnings from `huggingface_hub`.
    warnings.simplefilter(action='ignore', category=FutureWarning)

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map={"": args.device},
    )

    model = PeftModel.from_pretrained(
        model,
        args.model_path,
        device_map={"": args.device},
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    model = model.merge_and_unload()
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
