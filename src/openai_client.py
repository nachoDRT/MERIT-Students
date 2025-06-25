import os
from openai import OpenAI
from os.path import join, abspath, dirname
from typing import Dict
from utils import *
from PIL import Image
import io


class openaiClient:

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

    def generate_id_photo(self, b64_src: str) -> Image.Image:

        prompt = (
            "Passport-style photo: same person as in the reference, front-facing, "
            "neutral expression, shoulders visible, pure white background, studio lighting."
        )

        response = self.client.responses.create(
            model="gpt-4.1",
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{b64_src}",
                        },
                    ],
                }
            ],
            tools=[{"type": "image_generation"}],
        )

        image_generation_calls = [output for output in response.output if output.type == "image_generation_call"]

        image_data = [output.result for output in image_generation_calls]

        if image_data:
            image_base64 = image_data[0]
            # with open("gift-basket.png", "wb") as f:
            #     f.write(base64.b64decode(image_base64))
            return Image.open(io.BytesIO(base64.b64decode(image_base64))).convert("RGB")
        else:
            print(response.output.content)


def init_apis():
    secrets_path = join(dirname(dirname(abspath(__file__))), "config", "secrets.json")
    secrets = load_secrets(secrets_path)
    os.environ["OPENAI_API_KEY"] = secrets["openai"]


def load_secrets(file_path: str) -> Dict:
    return read_json(file_path)
