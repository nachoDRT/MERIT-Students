from datasets import Dataset, Features, Image, Value, DatasetDict
from huggingface_hub import HfApi, Repository, HfFolder
from io import BytesIO
from typing import Dict
from os.path import join, dirname, abspath
from utils import read_json


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
            "system": Value("string"),
        }
    )
    split = Dataset.from_dict(data["train"], features=features)

    # Create DatasetDict
    dataset = DatasetDict({"train": split})

    return dataset


def push_dataset_to_hf(dataset, repo_name):

    hf_config = get_hf_config()
    push_splits_to_hf(dataset, hf_config, repo_name)


def push_splits_to_hf(data: DatasetDict, configuration: Dict, repo_name: str):
    # Authentication
    HfFolder.save_token(configuration["hf_token"])

    api = HfApi()
    user = api.whoami()
    username = user["name"]

    for split, dataset in data.items():
        dataset_name = f"{username}/{repo_name}"
        dataset.push_to_hub(dataset_name, split=split, max_shard_size="500MB")


def get_hf_config() -> Dict:

    configuration_path = join(dirname(dirname(abspath(__file__))), "config", "secrets.json")
    configuration = read_json(configuration_path)

    return configuration
