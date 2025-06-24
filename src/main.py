import pandas as pd
from os.path import join, dirname, abspath
from datasets import load_dataset
from PIL import Image as PÌLIMage
import json


NUM_PROC = 8


def read_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def load_config():
    config_path = join(dirname(dirname(abspath(__file__))), "config", "config.json")
    return read_json(config_path)


config = load_config()
AGE_GROUPS = config["age_groups"]
ORIGINS = config["origins"]
GENDERS = config["genders"]


def get_merit_data() -> pd.DataFrame:

    blueprint_path = join(dirname(dirname(abspath(__file__))), "data", "dataset_blueprint.csv")

    try:

        df = pd.read_csv(blueprint_path)
        cols = ["file_name", "student_name", "student_gender", "student_name_origin"]
        relevant_columns = df[cols]

        return relevant_columns

    except FileNotFoundError:
        print(f"Merit dataset blueprint not found.")


def get_fair_face_subsets(decode=None):

    print("Loading Fair Face Dataset")
    fair_face_dataset = load_dataset("HuggingFaceM4/FairFace", "1.25", split="train")

    if decode:
        fair_face_dataset = fair_face_dataset.cast_column("image", PÌLIMage(decode=False))

    subsets = {}

    for age_id, age_tag in AGE_GROUPS.items():
        for race_id, race_tag in ORIGINS.items():
            for gender_id, gender_tag in GENDERS.items():
                key = f"{race_tag}_{gender_tag}_{age_tag}"
                subsets[key] = fair_face_dataset.filter(
                    lambda ex, a=age_id, r=race_id, g=gender_id: (
                        ex["age"] == a and ex["gender"] == g and ex["race"] == r
                    ),
                    num_proc=NUM_PROC,
                )

    return subsets


if __name__ == "__main__":

    merit_df = get_merit_data()
    fair_face_subsets = get_fair_face_subsets()

    east_asian_women_aged_20_29 = fair_face_subsets["east_asian_women_aged_20_29"]
    latino_men_aged_10_19 = fair_face_subsets["latino_men_aged_10_19"]
    print(f"Num of young_students_men: {len(east_asian_women_aged_20_29)}")
    print(f"Num of young_students_women: {len(latino_men_aged_10_19)}")
