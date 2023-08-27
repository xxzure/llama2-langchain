import boto3
import json
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
