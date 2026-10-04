import base64
import gzip
from pathlib import Path

DATASET_FILE = Path(r"d:\Project\voltforge\Voltforge_AI\training\voltforge_electronics_dataset.jsonl")
RUNNER_FILE = Path(r"d:\Project\voltforge\Voltforge_AI\training\colab_train_runner.py")

raw_data = open(DATASET_FILE, "rb").read()
compressed = gzip.compress(raw_data)
b64_str = base64.b64encode(compressed).decode("ascii")

part1 = """# ========================================================
# ⚡ VoltForge AI - High-Efficiency Model Fine-Tuning Pipeline
# ========================================================

# Step 1: Install Unsloth and required training packages
# (Note: xformers is omitted because PyTorch 2.x SDPA handles attention natively on T4 GPU)
print("--- [1/5] Installing Unsloth and Dependencies ---")
!pip install --upgrade --no-cache-dir unsloth unsloth_zoo trl peft accelerate bitsandbytes

# Step 2: Unpack embedded VoltForge Electronics Dataset (High-Density Verified Records)
print("--- [2/5] Preparing Dataset ---")
import base64, gzip, json
from datasets import Dataset

RAW_B64 = \"\"\""""

part2 = """\"\"\"
jsonl_bytes = gzip.decompress(base64.b64decode(RAW_B64))
records = [json.loads(line) for line in jsonl_bytes.decode('utf-8').splitlines() if line.strip()]
print(f"Successfully extracted {len(records)} high-density training records!")

# Step 3: Load Qwen 2.5 Coder 1.5B with 4-bit LoRA (Fresh base model for maximum precision)
print("--- [3/5] Loading Qwen 2.5 Coder 1.5B with High-Capacity LoRA ---")
from unsloth import FastLanguageModel
import torch

max_seq_length = 2048
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-Coder-1.5B-Instruct",
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
    return {"text": texts}

dataset = Dataset.from_list(records).map(formatting_prompts_func, batched=True)

# Step 4: Fine-tune model for Maximum Efficiency (Cosine Schedule, 300 steps)
print("--- [4/5] Starting High-Efficiency LoRA Fine-Tuning (~12 mins on T4 GPU) ---")
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=20,
        max_steps=300,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        output_dir="outputs",
    ),
)
trainer.train()

# Step 5: Export directly to Standalone Q4_K_M GGUF format
print("--- [5/5] Exporting directly to Standalone Q4_K_M GGUF ---")
model.save_pretrained_gguf(
    "voltforge_ai_trained",
    tokenizer,
    quantization_method="q4_k_m"
)
print("\\nSUCCESS! Model exported to voltforge_ai_trained/ directory!")
!ls -lh voltforge_ai_trained/

# Trigger automatic browser download
from google.colab import files
import os
for f in os.listdir('voltforge_ai_trained'):
    if f.endswith('.gguf'):
        full_path = os.path.join('voltforge_ai_trained', f)
        print(f"Triggering download for: {full_path}...")
        try:
            files.download(full_path)
        except Exception as e:
            print(f"Browser download notice: {e}. You can also download it from the left-side files panel in Colab!")
"""

full_content = part1 + b64_str + part2
with open(RUNNER_FILE, "w", encoding="utf-8") as f:
    f.write(full_content)

print(f"Successfully generated {RUNNER_FILE} (Length: {len(full_content)} bytes)")
