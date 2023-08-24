import boto3
import json
import vector_store
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from typing import Dict, List, Optional
from config import llm_model


class Llm(LLM):
    max_token: int = 10000
    temperature: float = 0.1
    top_p = 0.9
    history = []
    tokenizer: object = None
    model: object = None
    client: object = None

    def __init__(self):
        super().__init__()
        self.client = boto3.client(
            "sagemaker-runtime", region_name="us-east-1")

    @property
    def _llm_type(self) -> str:
        return "ChatLLM"

    def _call(self, prompt, stop: Optional[List[str]] = None) -> str:
        payload = {
            "inputs": prompt,
            "parameters": {"max_new_tokens": self.max_token, "top_p": self.top_p, "temperature": self.temperature}
        }
        print(payload)
        response = self.client.invoke_endpoint(
            EndpointName=llm_model,
            ContentType="application/json",
            Body=json.dumps(payload),
            CustomAttributes="accept_eula=true",
        )
        response = response["Body"].read().decode("utf8")
        response = json.loads(response)
        response = response[0]['generation']['content']

        if stop is not None:
            response = enforce_stop_tokens(response, stop)
        self.history = self.history + [[None, response]]

        return response


def llm_status():
    try:
        llm = Llm()
        llm._call([[{"role": "user", "content": "hello"}]])
        return """The initial model has loaded successfully and you can start a conversation"""
    except Exception as e:
        print(e)
        return """The model did not load successfully"""


llm = Llm()
vector_store = vector_store.get_vector_store("./data/advertising.json")


def build_llama2_prompt(messages):
    startPrompt = "<s>[INST] "
    endPrompt = " [/INST]"
    conversation = []
    for index, message in enumerate(messages):
        if message["role"] == "system" and index == 0:
            conversation.append(f"<<SYS>>\n{message['content']}\n<</SYS>>\n\n")
        elif message["role"] == "user":
            conversation.append(message["content"].strip())
        else:
            conversation.append(
                f" [/INST] {message['content'].strip()}</s><s>[INST] ")

    return startPrompt + "".join(conversation) + endPrompt


def get_knowledge_based_answer(query, top_k: int = 10, history_len: int = 3, temperature: float = 0.01, top_p: float = 0.1, history=[]):

    llm.temperature = temperature
    llm.top_p = top_p

    prompt_template = """
    {"role": "system", "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. There are some json about advertising content, based on these information, answer the user's questions concisely and professionally. If you can't get an answer from it, say 'The question cannot be answered based on known information' or 'Not enough relevant information is provided'."}
    {history}
    {"role": "user", "content":"Known content:{context}. Question:{question}"}"""
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["history", "context", "question"])
    llm.history = history[-history_len:] if history_len > 0 else []

    knowledge_chain = RetrievalQA.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(
            search_kwargs={"k": top_k}),
        prompt=prompt)
    knowledge_chain.combine_documents_chain.document_prompt = PromptTemplate(
        input_variables=["page_content"], template="{page_content}")

    knowledge_chain.return_source_documents = True

    result = knowledge_chain({"query": query})
    return result
