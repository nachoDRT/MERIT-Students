import os
from openai import OpenAI
from os.path import join, abspath, dirname
import traceback
import openai
import json

class openaiClient:

    def __init__(self, system_prompt):

        init_apis()
        self.client = OpenAI()
        self.system_prompt = system_prompt

    def sentiment_decide(self, user_prompt):

        model = "gpt-4.1"
        
        try:

            response = self.client.chat.completions.create(
                model=model,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": self.system_prompt
                    },
                    {
                    "role": "user",
                    "content": user_prompt,
                },
                ],
            )

            content = response.choices[0].message.content

            if not content:
                raise ValueError("OpenAI returned empty content")

            print(f"[judge] respuesta recibida, len={len(content)}")
            return content

        except openai.APIConnectionError as e:
            print(f"[judge] APIConnectionError: {e}")
            print(f"[judge] cause: {repr(e.__cause__)}")
            raise

        except openai.RateLimitError as e:
            print(f"[judge] RateLimitError: {e}")
            raise

        except openai.APIStatusError as e:
            print(f"[judge] APIStatusError: status={e.status_code}")
            print(f"[judge] request_id={getattr(e, 'request_id', None)}")
            print(f"[judge] response={e.response}")
            raise

        except Exception as e:
            print(f"[judge] Unexpected error: {type(e).__name__}: {e}")
            traceback.print_exc()
            raise


def init_apis():
    secrets_path = join(dirname(dirname(abspath(__file__))), "config", "secrets.json")
    secrets = load_secrets(secrets_path)
    os.environ["OPENAI_API_KEY"] = secrets["openai"]


def load_secrets(file_path: str):
    return read_json(file_path)


def read_json(path: str):
    with open(path, "r") as f:
        return json.load(f)