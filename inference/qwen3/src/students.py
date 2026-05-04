from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

if __name__ == "__main__":
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": None,
                },
                {
                    "type": "image",
                    "image": None,
                },
                {"type": "text", "text": "Describe both images."},
            ],
        }
    ]


    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-8B-Instruct", dtype="auto", device_map="auto"
    )

    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=1280)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text)