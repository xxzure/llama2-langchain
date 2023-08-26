import boto3
import json
import vector_store
from typing import Dict, List, Optional
from config import llm_model


class Llm():
    max_token: int = 10000
    temperature: float = 0.1
    top_p = 0.9
    history = []
    tokenizer: object = None
    model: object = None
    client: object = None

    def __init__(self):
        self.client = boto3.client(
            "sagemaker-runtime", region_name="us-east-1")

    def _call(self, prompt) -> str:
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

        return response


def llm_status():
    try:
        llm = Llm()
        llm._call([[{"role": "user", "content": "hello"}]])
        llm.history = []
        return """The initial model has loaded successfully and you can start a conversation"""
    except Exception as e:
        print(e)
        return """The model did not load successfully"""


llm = Llm()
vector_store = vector_store.get_vector_store("./data/advertising.json")


def build_llama2_prompt(system, user, history, history_len):
    messages = [system]
    length = len(history)
    if length > history_len:
        length = history_len
    for combined_message in history[-length:]:
        for message in combined_message:
            messages.append(message)
    messages.append(user)
    return [messages]


def get_knowledge_based_answer(query, top_k: int = 10, history_len: int = 3, temperature: float = 0.01, top_p: float = 0.1):

    llm.temperature = temperature
    llm.top_p = top_p

    system_prompt = {"role": "system", "content": "You are a helpful, respectful and honest assistant. There are some content about advertising, please answer the user's questions concisely and professionally. If you can't get an answer from it, return user 'The question cannot be answered based on known information' or 'Not enough relevant information is provided'."}

    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    document_list = retriever.get_relevant_documents(query)
    context = ""
    i = 1
    for doc in document_list:
        context += doc.page_content
        if i != len(document_list):
            context += ","
        i = i + 1
    user_prompt = {"role": "user",
                   "content": f"Known content: {context}.\nQuestion:{query}"}
    prompt = build_llama2_prompt(
        system_prompt, user_prompt, llm.history, history_len)
    response = llm._call(prompt)
    print(response)
    ai_prompt = {"role": "assistant", "content": response}
    llm.history.append([user_prompt, ai_prompt])
    return response


get_knowledge_based_answer("hello")
