import pandas as pd
from os.path import join, dirname, abspath
from datasets import load_dataset
from PIL import Image as PÌLImage
from utils import *
from utils_hf import *
from openai_client import openaiClient
import tqdm as tqdm


NUM_PROC = 8
AGE_PRIORITY = ["10_to_19", "20_to_29", "30_to_39"]

config = load_config("config_fairface.json")
AGE_GROUPS = config["age_groups"]
ORIGINS = config["origins"]
GENDERS = config["genders"]


def get_merit_data() -> pd.DataFrame:

    blueprint_path = join(dirname(dirname(abspath(__file__))), "data", "dataset_blueprint.csv")

    try:

        df = pd.read_csv(blueprint_path)
        cols = [
            "file_name",
            "language",
            "student_name",
            "student_index",
            "student_gender",
            "student_name_origin",
            "average_grade",
        ]
        relevant_columns = df[cols]
        relevant_columns = one_student_per_row(relevant_columns)

        return relevant_columns

    except FileNotFoundError:
        print(f"Merit dataset blueprint not found.")


def get_fair_face_subsets(decode=False):

    print("Loading Fair Face Dataset")
    fair_face_dataset = load_dataset("HuggingFaceM4/FairFace", "1.25", split="train")

    if decode:
        fair_face_dataset = fair_face_dataset.cast_column("image", PÌLImage(decode=False))

    subsets = {}

    for age_tag, age_id in AGE_GROUPS.items():
        for origin_tag, origin_id in ORIGINS.items():
            for gender_tag, gender_id in GENDERS.items():
                key = f"{origin_tag}_{gender_tag}_{age_tag}"

                """ As the authors of this dataset, we prefer to use the term "origin", 
                but "race" is used in Fair Face dataset. Thus we use it to access that column in the df"""
                subsets[key] = fair_face_dataset.filter(
                    lambda ex, a=age_id, r=origin_id, g=gender_id: (
                        ex["age"] == a and ex["gender"] == g and ex["race"] == r
                    ),
                    num_proc=NUM_PROC,
                )

    return subsets


def link_student_to_id_pic(merit_df: pd.DataFrame, fair_face_subsets: dict):

    english_map = load_config("config_merit.json")["english_origin_map"]
    spanish_map = load_config("config_merit.json")["spanish_origin_map"]

    discriminator = openaiClient()

    for i, row in enumerate(merit_df.itertuples(index=False)):
        ff_origin = map_origin(row.language, row.student_name_origin, english_map, spanish_map)

        name = row.file_name
        index = row.student_index
        gender = row.student_gender
        name_origin = row.student_name_origin
        grade = row.average_grade

        subset, used_age_group = get_subset_and_age_group(fair_face_subsets, ff_origin, gender, AGE_PRIORITY)

        taken, remaining = take_first(subset)
        fair_face_subsets[f"{ff_origin}_{gender}_{used_age_group}"] = remaining

        ff_image = taken[0]["image"]
        ff_image.show()

        ff_encoded = encode_image(ff_image)
        decision, reason = discriminator.process_image(ff_encoded)
        print(f"Image {i}. {decision}: {reason}. Group: {ff_origin}_{gender}_{used_age_group}")

        # new_img = discriminator.generate_id_photo(ff_encoded)
        # new_img.show()

        if i > 10:
            break


if __name__ == "__main__":

    merit_df = get_merit_data()
    fair_face_subsets = get_fair_face_subsets()

    link_student_to_id_pic(merit_df, fair_face_subsets)
