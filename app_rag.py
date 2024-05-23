import argparse
import sys
import warnings
from multiprocessing.pool import ThreadPool
from operator import itemgetter

import streamlit as st
import torch
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from langchain_community.vectorstores import Chroma
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
    pipeline,
    TextIteratorStreamer,
)


def format_docs(docs):
    document = "".join([d.metadata['content'] for d in docs])
    return {'document': document}


@st.cache_resource()
def prepare(model_path: str, base_model: str, lora: bool, db_path: str, embedding_model: str):
    # Disable FutureWarnings from `huggingface_hub`.
    warnings.simplefilter(action='ignore', category=FutureWarning)

    device = torch.device('cuda:0')

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path if not lora else base_model,
        quantization_config=bnb_config,
        device_map="auto",
    )

    if lora:
        model = PeftModel.from_pretrained(
            model,
            model_path,
            device_map="auto",
        )

    tokenizer = AutoTokenizer.from_pretrained(base_model)

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
            max_new_tokens=128,
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

    embedding = HuggingFaceEmbeddings(
        model_name=embedding_model, model_kwargs={'device': device})
    db = Chroma(persist_directory=db_path, embedding_function=embedding)
    retriever = db.as_retriever(search_kwargs={'k': 1})

    chain = retriever | format_docs | prompt | hf

    return chain, streamer


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
    parser.add_argument('--embedding-model', type=str, default="sentence-transformers/distiluse-base-multilingual-cased-v2",
                        help="HuggingFace embedding model name.")
    parser.add_argument('--db', type=str, default="./db",
                        help="Path to the directory to save the vector database.")
    args = parser.parse_args()

    chain, streamer = prepare(
        args.model_path, args.base_model, args.lora, args.db, args.embedding_model)

    st.title('Simple Retrieval-Augmented Generation (RAG) Demo')
    with st.form('form'), ThreadPool(processes=1) as pool:
        query = st.text_input(
            '搜尋：',
            value="大地震引發海嘯",
            help='Enter the query to search for relevant news.')
        submitted = st.form_submit_button('送出')
        if submitted:
            # Streaming output
            # async_result = pool.apply_async(chain.invoke, args=(query,))
            # st.write_stream(streamer)
            # result = async_result.get()

            # Sync output
            result = chain.invoke(query)
            st.write(result)

            # result = chain.invoke(query)
            # st.write(result['summarization'])
            # st.write("---\n"
            #          "#### 以下為擷取的新聞文章")
            # for document in result['documents']:
            #     st.write(
            #         f"- ##### {document.metadata['title']}\n\n"
            #         f"\t{document.metadata['date']}\n\n"
            #         f"\t{document.metadata['content']}\n\n"
            #         f"\t[source]({document.metadata['url']})\n\n")


if __name__ == '__main__':
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    main()
