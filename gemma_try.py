# pip install accelerate
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import torch
torch.manual_seed(0)
model_id = "google/gemma-2-9b"
def func():
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(
        pretrained_model_name_or_path=model_id,
        token="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",
        device_map="auto",
        torch_dtype=torch.bfloat16
    )

    input_text = "Write me a poem about Machine Learning."
    input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")
    model_outputs = model(**input_ids).last_hidden_state
    embeddings = torch.prod(model_outputs.squeeze(),dim=0)
    return embeddings
# print(tokenizer.decode(outputs[0]))
