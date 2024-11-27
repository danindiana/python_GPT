import torch
from colpali_engine.models import ColQwen2, ColQwen2Processor

def load_model_and_processor(device, model_name="vidore/colqwen2-v0.1"):
    model = ColQwen2.from_pretrained(
        model_name, torch_dtype=torch.float16
    ).to(device).eval()
    processor = ColQwen2Processor.from_pretrained(model_name)
    return model, processor
