from datasets import Dataset, Features, Image, Value, DatasetDict
from huggingface_hub import HfApi, Repository, HfFolder
from io import BytesIO
from typing import Dict


def format_data(data: Dict) -> DatasetDict:
    # Convert to HF dataset
    features = Features(
        {
            "id_image": Image(),
            "student_name": Value("string"),
            "student_index": Value("int32"),
            "student_gender": Value("string"),
            "student_name_origin": Value("string"),
            "average_grade": Value("float32"),
        }
    )
    split = Dataset.from_dict(data["train"], features=features)

    # Create DatasetDict
    dataset = DatasetDict({"train": split})

    return dataset
