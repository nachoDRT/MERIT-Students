import json
from os.path import join, dirname, abspath
from io import BytesIO
import base64
from typing import List, Dict
import pandas as pd
import ast


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


def take_first(dataset, n=1):
    size = len(dataset)
    n = min(n, size)

    taken = dataset.select(range(n))
    if n == size:
        remaining = dataset.select([])
    else:
        remaining = dataset.select(range(n, size))

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


def unpack_singleton(value, cast=float):
    if isinstance(value, list):
        return cast(value[0])

    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return cast(parsed[0])
        except (ValueError, SyntaxError):
            pass

    return cast(value)


def one_student_per_row(merit_df: pd.DataFrame):
    df = merit_df.copy()

    df["average_grade"] = df["average_grade"].apply(unpack_singleton)

    agg_funcs = {col: "first" for col in df.columns if col != "average_grade"}
    agg_funcs["average_grade"] = "mean"

    students_df = df.groupby("student_name", as_index=False).agg(agg_funcs)

    students_df = students_df.sort_values("student_index", ascending=True).reset_index(drop=True)

    return students_df
