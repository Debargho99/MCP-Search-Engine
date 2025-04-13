from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

def load_model_pipeline(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map='auto'  # for GPU-based loading
    )
    return pipeline("text-generation", model=model, tokenizer=tokenizer)
