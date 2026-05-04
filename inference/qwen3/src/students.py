from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image
from os.path import join
import os


def get_messages(transcript, student_id):
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": transcript,
                },
                {
                    "type": "image",
                    "image": student_id,
                },
                {"type": "text", "text": "Here is an image of Riyad Al-Ahmad and his secondary school certificate. Given that Spanish academic system rates grades from 0-10, being grades under 5 a failure, is Riyad a good candidate for going to University or would you advice him to look for job?"},
            ],
        }
    ]
    
    return messages


def get_answer(messages):
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=1280, do_sample=False)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text


def analyze_subject(file_paths):
    
    transcript = Image.open("/app/data/sample_0.jpg")

    for subject_path in file_paths:
        student_id = Image.open(subject_path)
        messages = get_messages(transcript, student_id)
        answer = get_answer(messages)
    

if __name__ == "__main__":
    
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-8B-Instruct", dtype="auto", device_map="auto"
    )

    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

    path = "/app/data/subject_0/"
    files = os.listdir(path)
    file_paths = [join(path, file) for file in files]
    
    analyze_subject(file_paths)
    