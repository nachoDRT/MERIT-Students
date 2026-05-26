import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

_MODEL_REGISTRY = {
    'qwen3-vl-8b-instruct': 'Qwen/Qwen3-VL-8B-Instruct',
}


def load_model_visual(model_name, device='cuda', cache_dir=None):
    """Load Qwen3-VL with AutoProcessor for multimodal (image+text) inference."""
    if model_name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: '{model_name}'. Registered: {list(_MODEL_REGISTRY.keys())}"
        )
    model_id  = _MODEL_REGISTRY[model_name]
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, cache_dir=cache_dir)
    model     = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, dtype=torch.bfloat16, cache_dir=cache_dir,
    )
    model.eval()
    return model, processor
