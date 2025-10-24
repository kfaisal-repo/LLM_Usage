from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

checkpoint = "Qwen/Qwen2.5-0.5B"
device = "cpu"  # or "cuda" if you have a GPU

# optional: speed tweaks for CPU (set threads if desired)
torch.set_num_threads(4)  # tune to your machine

# load tokenizer (set trust_remote_code=True if model uses custom tokenizer)
tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)

# load model with some memory-sparing option (if supported)
model = AutoModelForCausalLM.from_pretrained(checkpoint, low_cpu_mem_usage=True, trust_remote_code=True)
model.to(device)

# use a real prompt instead of empty string
prompt = "Write a poem on spring season."
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# generation parameters you should tune
gen_kwargs = dict(
    max_new_tokens=256,
    do_sample=True,
    temperature=0.8,
    top_p=0.95,
    eos_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.sep_token_id,
    pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id,
)

outputs = model.generate(**inputs, **gen_kwargs)
text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(text)
