import os
from openai import OpenAI
from os.path import join, abspath, dirname
from typing import Dict
from utils import *


class openaiDiscriminator:

    def __init__(self):

        init_apis()
        self.client = OpenAI()

    def process_image(self, base64_image):

        model = "gpt-4o"

        response = self.client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an administrative assistant who decides whether a student's photo is adequate "
                        "for school registration. Reply ONLY with a JSON object containing exactly two keys: "
                        "'decision' (boolean) and 'reason' (string)."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Look at the image and decide if it meets all the following:\n"
                                "1. The student appears **alone** in the picture.\n"
                                "2. The face is fully recognizable (no sunglasses, masks, or heavy shadows).\n"
                                "3. Background uniformity is NOT required; noise is allowed.\n"
                                "Return **only** this JSON schema:\n"
                                "{\n"
                                '  "decision": true | false,\n'
                                '  "reason": "<short explanation>"\n'
                                "}"
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                    ],
                },
            ],
        )

        data = json.loads(response.choices[0].message.content)
        decision_bool = bool(data["decision"])
        reason_text = data["reason"]

        return decision_bool, reason_text


def init_apis():
    secrets_path = join(dirname(dirname(abspath(__file__))), "config", "secrets.json")
    secrets = load_secrets(secrets_path)
    os.environ["OPENAI_API_KEY"] = secrets["openai"]


def load_secrets(file_path: str) -> Dict:
    return read_json(file_path)
