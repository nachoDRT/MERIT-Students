import json
from os.path import join, dirname, abspath
from io import BytesIO
import base64


def read_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def load_config(file: str):
    config_path = join(dirname(dirname(abspath(__file__))), "config", file)
    return read_json(config_path)


def map_origin(system: str, name_origin: str, english_map: dict, spanish_map: dict):

    if system == "english":
        return english_map[name_origin]
    elif system == "spanish":
        return spanish_map[name_origin]
    else:
        raise ValueError("System option not implemented.")


def encode_image(img):

    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()

    return base64.b64encode(img_bytes).decode("utf-8")
