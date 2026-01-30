import json
import random
import torch
import gc
import os
from torch.utils.data import Dataset
from transformers import (
    Trainer, 
    TrainingArguments, 
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training, 
    PeftModel
)

# CONFIGURATION
OUTPUT_DIR = "Output_dir"
BASE_MODEL = "Base_model"
EPOCHS = 15  # Reduced from 20; LoRA learns faster, and 20 might overfit small data
TRAIN_FILE = "augmented_train_data.txt"
INTENT_FILE = "intent.json"
DB_FILE = "data.json"

class RouterDataset(Dataset):
    def __init__(self, txt_file, tokenizer, max_length=128): # Increased slightly for safety
        self.tokenizer = tokenizer
        self.max_length = max_length
        print(f"📂 Reading lines from {txt_file}...")
        with open(txt_file, "r", encoding="utf-8") as f:
            self.lines = [line.strip() for line in f.readlines() if line.strip()]
        print(f"✅ Loaded {len(self.lines)} lines.")

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        encodings = self.tokenizer(
            line, truncation=True, max_length=self.max_length, padding="max_length"
        )
        return {
            'input_ids': torch.tensor(encodings['input_ids']),
            'attention_mask': torch.tensor(encodings['attention_mask']),
            'labels': torch.tensor(encodings['input_ids'])
        }

def build_training_data():
    # ... (Keep your original build_training_data function here)
    # It works perfectly for generating the .txt file.
    return TRAIN_FILE

def train(filename):
    print(f"🚀 Loading model in 4-bit Quantization: {BASE_MODEL}...")
    
    # 1. Quantization Config: Reduces 1B model VRAM from ~2GB to ~400MB
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4", # NF4 is the "sweet spot" for accuracy
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
    )
    
    # 2. Prepare for PEFT (LoRA)
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=16, # Rank: controls the number of trainable parameters
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters() # Should show < 2% of total params

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = RouterDataset(filename, tokenizer)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,

                num_train_epochs=EPOCHS,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16, # Higher steps to compensate for batch size 1
        learning_rate=2e-4, # LoRA needs a higher LR than full fine-tuning
        fp16=True,
        optim="paged_adamw_32bit", # Prevents OOM by paging to system RAM
        gradient_checkpointing=True, # Critical for 4GB VRAM
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        report_to="none",
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
    
    print("🔥 Starting QLoRA fine-tuning...")
    trainer.train()
    
    # Save only the LoRA adapters (very small file)
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✅ Adapters saved to {OUTPUT_DIR}")

def test():
    print("\n\n🧪 LOADING MODEL + ADAPTERS FOR TESTING...")
    
    # 1. Load the Base Model (again in 4-bit to save VRAM)
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb_config, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    
    # 2. Load and Merge the LoRA Adapters
    model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
    model.eval()

    prompts = ["alias names of Sinhagad fort", "where is Parvati located?"]
    
    for txt in prompts:
        prompt = f"User: {txt} -> Bot:"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_state():
            outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
        
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\n💬 You: {txt}\n🤖 Bot: {decoded.split('Bot:')[-1].strip()}")

if __name__ == "__main__":
    generated_file = build_training_data()
    train(generated_file)
    test()
