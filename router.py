import json
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

#  CONFIGURATION
MODEL_PATH = "./Qwen2.5-0.5B-Router/checkpoint-600"  # <<< UPDATED to new model directory
DB_FILE = "data.json"

#  HELPER FUNCTIONS
def load_resources():
    """Loads the fine-tuned model, tokenizer, and JSON database."""
    print(f"🔄 Loading model from {MODEL_PATH}...")
    try:
        # Load the tokenizer and model from the specified path
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
        
        # Load the knowledge base
        with open(DB_FILE, "r", encoding="utf-8") as f:
            db = json.load(f)
            
        print("✅ System Ready!")
        return tokenizer, model, db
    except Exception as e:
        print(f"❌ Critical Error: Could not load resources. {e}")
        print("Please ensure the model has been trained and the paths are correct.")
        exit()

def normalize_entity(raw_name):
    """
    Translates various user inputs or model outputs into a canonical database key.
    This makes the lookup process robust to spelling variations or nicknames.
    """
    if not raw_name: return None
    n = raw_name.lower().strip("'\" -")
    
    if "rajgad" in n or "raja" in n or "murumbdev" in n:
        return "Rajgad Fort"
    if "torna" in n or "prachandagad" in n:
        return "Torna Fort"
    if "parvati" in n or "tekdi" in n:
        return "Parvati Hill"
        
    return None

def parse_command(text):
    """
    A more robust parser for the model's output.
    It splits the command by '|' and processes key-value pairs.
    Example: "INTENT:GET_FACT | ENTITY:Rajgad Fort | ATTRIBUTE:elevation"
    """
    parts = {"INTENT": None, "ENTITY": None, "ATTRIBUTE": None}
    
    # Split the command string by the pipe delimiter
    segments = [s.strip() for s in text.split('|')]
    
    for seg in segments:
        # Split each segment into a key and a value
        try:
            key, value = [p.strip() for p in seg.split(':', 1)]
            
            # Update the dictionary with the parsed key-value pair
            if key.upper() in parts:
                parts[key.upper()] = value
        except ValueError:
            # Handle cases where a segment doesn't contain a ':'
            print(f"⚠️ Warning: Could not parse segment: '{seg}'")

    # Normalize the extracted entity name
    if parts["ENTITY"]:
        parts["ENTITY"] = normalize_entity(parts["ENTITY"])
        
    return parts

# MAIN BOT RESPONSE LOGIC
def get_bot_response(tokenizer, model, db, user_input):
    """
    Generates a response by querying the model, parsing its output, 
    and looking up the result in the database.
    """
    # 1. Format the prompt exactly as the model was trained
    prompt = f"User: {user_input} -> Bot:"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # 2. Generate the command from the model
    outputs = model.generate(
        **inputs, 
        max_new_tokens=60, 
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Isolate the model's command
    if "Bot:" in full_output:
        cmd = full_output.split("Bot:")[-1].strip()
    else:
        cmd = full_output.replace(f"User: {user_input} ->", "").strip()

    print(f"🤖 RAW: {cmd}") 

    # 3. Parse the command to get structured data
    parts = parse_command(cmd)
    
    if parts["ENTITY"]:
        print(f"   ↳ TRANSLATED: {parts['ENTITY']} | ATTRIBUTE: {parts.get('ATTRIBUTE')}")
    
    intent = parts["INTENT"]
    entity = parts["ENTITY"]
    attr = parts["ATTRIBUTE"]
    
    # 4. Database Lookup
    if not entity or entity not in db:
        return "I'm sorry, I don't recognize that name. Could you be more specific?"
        
    entry = db[entity]
    
    if intent == "GET_SUMMARY":
        summary = entry.get('history', {}).get('significance', 'It is a place of historical importance.')
        return f"SUMMARY: {entity} is located in {entry.get('location', 'an unknown location')}. {summary}"
        
    if not attr:
         return f"I have information about {entity}. What would you like to know? For example, its location, height, or history."

    # Traverse the DB to find the requested attribute
    val = "I don't have information on that specific attribute."
    current_level = entry
    path = attr.split('.')
    
    try:
        for key in path:
            current_level = current_level[key]
        val = current_level
    except (KeyError, TypeError):
        # This will trigger if a key is not found or if we try to index a non-dict
        pass

    if isinstance(val, list):
        return f"The {attr.replace('_', ' ')} of {entity} are: " + ", ".join(map(str, val))
        
    return f"FACT: The {attr.replace('_', ' ')} of {entity} is: {val}"

#  RUNTIME LOOP
if __name__ == "__main__":
    tokenizer, model, db = load_resources()
    print("-" * 50)
    print("      🏰 Fort Guide Bot v2.0 🏰")
    print(" (Powered by an improved fine-tuned model)")
    print("-" * 50)
    
    while True:
        q = input("\nYou: ")
        if q.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        print(f"Bot: {get_bot_response(tokenizer, model, db, q)}")
