import boto3
import json
from config import llm_model
from langchain.llms.base import LLM

from threading import Thread
from typing import Optional

from langchain import PromptTemplate, LLMChain
from langchain.llms.base import LLM
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


def init_chain(model, tokenizer):
    class CustomLLM(LLM):

        """Streamer Object"""

        streamer: Optional[TextIteratorStreamer] = None

        def _call(self, prompt, stop=None, run_manager=None) -> str:
            self.streamer = TextIteratorStreamer(
                tokenizer, skip_prompt=True, Timeout=5)
            inputs = tokenizer(prompt, return_tensors="pt")
            kwargs = dict(
                input_ids=inputs["input_ids"], streamer=self.streamer, max_new_tokens=20)
            thread = Thread(target=model.generate, kwargs=kwargs)
            thread.start()
            return ""

        @property
        def _llm_type(self) -> str:
            return "custom"

    llm = CustomLLM()

    template = """Question: {question}
    Answer: Let's think step by step."""
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    return llm_chain, llm


class Llm():
    max_token: int = 2048
    temperature: float = 0.1
    top_p = 0.9
    history = []
    tokenizer: object = None
    model: object = None
    client: object = None

    def __init__(self):
        # self.client = boto3.client("sagemaker-runtime", region_name="us-east-1")
        self.client = boto3.client('bedrock', 'us-east-1')

    def _call(self, prompt, stop=None, run_manager=None) -> str:
        # payload = {
        #     "inputs": prompt,
        #     "parameters": {"max_new_tokens": self.max_token, "top_p": self.top_p, "temperature": self.temperature}
        # }
        # print(payload)
        # response = self.client.invoke_endpoint(
        #     EndpointName=llm_model,
        #     ContentType="application/json",
        #     Body=json.dumps(payload),
        #     CustomAttributes="accept_eula=true",
        # )
        # response = response["Body"].read().decode("utf8")
        # response = json.loads(response)
        # response = response[0]['generation']['content']
        payload = {
            "prompt": prompt,
            "max_tokens_to_sample": self.max_token,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": 250,
            "stop_sequences": [],
        }
        print(json.dumps(payload))
        accept = 'application/json'
        contentType = 'application/json'
        response = self.client.invoke_model(
            modelId='anthropic.claude-v2',
            body=json.dumps(payload),
            accept=accept,
            contentType=contentType
        )
        response_body = json.loads(response.get('body').read())

        print(response_body)
        # stream = response.get('body')
        # print(stream)
        # if stream:
        #     for event in stream:
        #         print(event)
        #         chunk = event.get('chunk')
        #         if chunk:
        #             response = json.loads(chunk.get('bytes').decode())
        #             print(response)
        return response_body.get('completion')
