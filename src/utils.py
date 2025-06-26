import json
from os.path import join, dirname, abspath
from io import BytesIO
import base64
from typing import List, Dict
import pandas as pd


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


def take_first(dataset):
    taken = dataset.select(range(1))
    remaining = dataset.select(range(1, len(dataset)))

    return taken, remaining


def get_subset_and_age_group(ff_subsets: Dict, ff_origin: str, gender: str, age_priority: List):
    subset = None
    used_age_group = None

    for age_group in age_priority:
        key = f"{ff_origin}_{gender}_{age_group}"
        candidate = ff_subsets.get(key)
        if candidate and len(candidate):
            subset = candidate
            used_age_group = age_group
            break

    if subset is None:
        print(f"No remaining images for {ff_origin} {gender}")
        return None, None

    return subset, used_age_group


def one_student_per_row(merit_df: pd.DataFrame):
    merit_df["average_grade"] = merit_df["average_grade"].str[0]

    agg_funcs = {col: "first" for col in merit_df.columns if col != "average_grade"}

    agg_funcs["average_grade"] = "mean"

    students_df = merit_df.groupby("student_name", as_index=False).agg(agg_funcs)

    return students_df
