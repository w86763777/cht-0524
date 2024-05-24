import argparse
import random
import warnings
from multiprocessing.pool import ThreadPool
from threading import Thread

import streamlit as st
import torch
from datasets import Dataset
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
    pipeline,
    TextIteratorStreamer,
)


@st.cache_resource()
def prepare(model_path: str, base_model: str, lora: bool):
    # Disable FutureWarnings from `huggingface_hub`.
    warnings.simplefilter(action='ignore', category=FutureWarning)

    dataset = Dataset.from_csv('./dataset.csv')
    # set seed for reproducibility
    datasets = dataset.train_test_split(test_size=0.2, seed=0)
    dataset = datasets['test']
    dataset = dataset.filter(lambda x: len(x['document']) < 200)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path if not lora else base_model,
        quantization_config=bnb_config,
        device_map={"": 0},
    )

    if lora:
        model = PeftModel.from_pretrained(
            model,
            model_path,
            device_map={"": 0},
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True)
    generator = pipeline(
        task='text-generation',
        model=model,
        tokenizer=tokenizer,
        streamer=streamer,
        generation_config=GenerationConfig(
            max_new_tokens=256,
            do_sample=False,
        ),
        return_full_text=False,
    )
    hf = HuggingFacePipeline(pipeline=generator)

    template = (
        "以下會呈現一篇新聞文章，請為這篇文章寫一段一句話左右的摘要：\n\n"
        "文章：{document}\n\n"
        "摘要："
    )
    prompt = PromptTemplate.from_template(template)

    chain = prompt | hf

    return chain, streamer, dataset


def generate_response(chain, streamer, document):
    thread = Thread(target=chain.invoke, kwargs={'input': {'document': document}})
    thread.start()
    for c in streamer:
        yield c
    thread.join()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str,
                        help='Path to the model weights directory. If the '
                             '--lora flag is set, this should be the path to '
                             'the LoRA weights directory.')
    parser.add_argument('--base-model', type=str, default="meta-llama/Llama-2-7b-hf",
                        help='Model name or path to the base model.')
    parser.add_argument('--lora', action='store_true', default=False,
                        help='See --model flag for details.')
    args = parser.parse_args()
    chain, streamer, test_dataset = prepare(
        args.model_path, args.base_model, args.lora)

    st.title('新聞摘要產生器')
    with st.form('form'), ThreadPool(processes=1) as pool:
        random_news = st.form_submit_button('隨機新聞')
        if random_news:
            idx = random.randint(0, len(test_dataset) - 1)
            news = test_dataset[idx]['document']
        else:
            news = st.session_state.get('news', "")
        input_news = st.text_area(
            '新聞內容：',
            value=news,
            help='輸入新聞內容',
            height=200,
        )
        st.session_state['news'] = input_news
        submitted = st.form_submit_button('送出')
        if submitted:
            # Streaming output
            # async_result = pool.apply_async(chain.invoke, args=(input_news,))
            # st.write_stream(streamer)
            # async_result.wait()

            # Sync output
            st.write(chain.invoke(input_news))


if __name__ == '__main__':
    main()
