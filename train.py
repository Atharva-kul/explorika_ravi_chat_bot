import json
import os
import random
import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling


# CONFIGURATION

OUTPUT_DIR = "./gpt2-router-vfixes"  
BASE_MODEL = "./my_gpt2"
EPOCHS = 10


class RouterDataset(Dataset):
    def __init__(self, txt_file, tokenizer, max_length=64):
        self.tokenizer = tokenizer
        self.examples = []
        
        print(f"📂 Reading {txt_file}...")
        with open(txt_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        for line in lines:
            
            encodings = tokenizer(
                line, 
                truncation=True, 
                max_length=max_length, 
                padding="max_length"
            )
            
            item = {
                'input_ids': torch.tensor(encodings['input_ids']),
                'attention_mask': torch.tensor(encodings['attention_mask']),
                'labels': torch.tensor(encodings['input_ids'])
            }
            self.examples.append(item)
            
        print(f"✅ Loaded {len(self.examples)} clean examples.")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def generate_data():
    print("⚙️ Generating Training Data...")
    
    # We use a special separator " -> " to make it super clear for the model
    patterns = [
        # Location
        ("where is {ent}", "INTENT:GET_FACT | ENTITY:{db} | ATTRIBUTE:location"),
        ("location of {ent}", "INTENT:GET_FACT | ENTITY:{db} | ATTRIBUTE:location"),
        
        # Elevation
        ("height of {ent}", "INTENT:GET_FACT | ENTITY:{db} | ATTRIBUTE:elevation"),
        ("how tall is {ent}", "INTENT:GET_FACT | ENTITY:{db} | ATTRIBUTE:elevation"),
        
        # Builder
        ("who built {ent}", "INTENT:GET_FACT | ENTITY:{db} | ATTRIBUTE:history.built_by"),
        ("founder of {ent}", "INTENT:GET_FACT | ENTITY:{db} | ATTRIBUTE:history.built_by"),
        
        # Summary
        ("tell me about {ent}", "INTENT:GET_SUMMARY | ENTITY:{db}"),
        ("summary of {ent}", "INTENT:GET_SUMMARY | ENTITY:{db}"),
    ]
    
    # 2. ENTITIES (The Knowledge)
    entities = [
        ("Rajgad", "Rajgad Fort"), ("Raja Fort", "Rajgad Fort"),
        ("Torna", "Torna Fort"), ("Prachandagad", "Torna Fort"),
        ("Parvati", "Parvati Hill"), ("Para Hill", "Parvati Hill"),
    ]

    lines = []
    # Generate 1000 examples
    for _ in range(1000):
        tmpl, cmd_tmpl = random.choice(patterns)
        user_ent, db_ent = random.choice(entities)
        
        user_text = tmpl.format(ent=user_ent)
        bot_cmd = cmd_tmpl.format(db=db_ent)
        
        # Format: "User: [Question] -> Bot: [Command]"
        # The '->' arrow helps the model distinguish input from output
        line = f"User: {user_text} -> Bot: {bot_cmd}"
        lines.append(line)

    filename = "train_custom.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        
    return filename

# main training loop
def train(filename):
    print(f" Loading {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    
    tokenizer.pad_token = tokenizer.eos_token

    # USE OUR CUSTOM CLASS
    dataset = RouterDataset(filename, tokenizer)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=8,
        save_steps=500,
        learning_rate=5e-5,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    print(" Starting Custom Training...")
    trainer.train()
    
    print(f" Saving model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


def test():
    print("\n TESTING...")
    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
    model = AutoModelForCausalLM.from_pretrained(OUTPUT_DIR)
    
    prompts = ["where is Rajgad?", "height of Torna", "who built Parvati?"]
    
    for txt in prompts:
        # Prompt must match training format EXACTLY (User: ... -> Bot:)
        prompt = f"User: {txt} -> Bot:"
        
        inputs = tokenizer(prompt, return_tensors="pt")
        
        outputs = model.generate(
            **inputs, 
            max_new_tokens=40, 
            do_sample=False,        # Greedy
            pad_token_id=tokenizer.eos_token_id
        )
        
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract command
        if "Bot:" in decoded:
            cmd = decoded.split("Bot:")[-1].strip()
            print(f"In:  {txt}")
            print(f"Out: {cmd}")
            print("-" * 20)

if __name__ == "__main__":
    f = generate_data()
    train(f)
    test()