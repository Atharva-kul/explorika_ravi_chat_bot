import json
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM


#  CONFIGURATION
MODEL_PATH = "./gpt2-router-vfixes"
DB_FILE = "data.json"


#  HELPER FUNCTIONS
def load_resources():
    print(f"Loading model from {MODEL_PATH}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
        
        with open(DB_FILE, "r", encoding="utf-8") as f:
            db = json.load(f)
            
        print(" System Ready!")
        return tokenizer, model, db
    except Exception as e:
        print(f" Error: {e}")
        exit()

def normalize_entity(raw_name):
    """
    Translates model hallucinations/nicknames into strict DB keys.
    """
    if not raw_name: return None
    n = raw_name.lower().strip("' -") 
    
    # RAJGAD MAPPER
    if "raj" in n or "raja" in n: return "Rajgad Fort"
    
    # TORNA MAPPER
    if "torn" in n or "prach" in n: return "Torna Fort"
        
    # PARVATI MAPPER
    if "par" in n or "para" in n: return "Parvati Hill"
        
    return None

def parse_command_smart(text):
    parts = {"INTENT": "GET_FACT", "ENTITY": None, "ATTRIBUTE": None}
    
    # 1. PARSE INTENT
    if "SUMMARY" in text: parts["INTENT"] = "GET_SUMMARY"
    elif "FACT" in text: parts["INTENT"] = "GET_FACT"
        
    # 2. PARSE ENTITY
    # Regex: Look for ENTITY, eat any separator (:-_ space), eat quotes, capture text
    ent_match = re.search(r"ENTITY\s*[:_\-\s]*['\"]?([a-zA-Z\s]+)", text, re.IGNORECASE)
    if ent_match:
        parts["ENTITY"] = normalize_entity(ent_match.group(1))

    # 3. PARSE ATTRIBUTE 
    # Regex: Look for ATTRIBUTE, eat ANY separator (colon, hyphen, underscore, space), eat quotes
    # This handles "ATTRIBUTE:-elevation" or "ATTRIBUTE_elevation"
    attr_match = re.search(r"ATTRIBUTE\s*[:_\-\s]*['\"]?([a-zA-Z._]+)", text, re.IGNORECASE)
    if attr_match:
        parts["ATTRIBUTE"] = attr_match.group(1).strip()

    return parts

# MAIN BOT RESPONSE LOGIC
def get_bot_response(tokenizer, model, db, user_input):
    # 1. EXACT PROMPT MATCH
    prompt = f"User: {user_input} -> Bot:"
    
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # 2. GENERATE
    outputs = model.generate(
        **inputs, 
        max_new_tokens=60, 
        do_sample=False,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "Bot:" in full_output:
        cmd = full_output.split("Bot:")[-1].strip()
    else:
        cmd = full_output.strip()

    print(f" RAW: {cmd}") 

    # 3. SMART PARSING
    parts = parse_command_smart(cmd)
    
    if parts["ENTITY"]:
        print(f"   ↳ TRANSLATED: {parts['ENTITY']} | ATTRIBUTE: {parts['ATTRIBUTE']}")
    
    intent = parts["INTENT"]
    entity = parts["ENTITY"]
    attr = parts["ATTRIBUTE"]
    
    # 4. DATABASE LOOKUP
    if not entity or entity not in db:
        return "I didn't recognize that fort name."
        
    entry = db[entity]
    
    if intent == "GET_SUMMARY":
        return f"SUMMARY: {entity} is in {entry['location']}. {entry['history'].get('significance', '')}"
        
    if not attr:
         return f"I know about {entity}. It is located at {entry['location']}."

    val = "Unknown"
    
    # Try exact match
    if attr in entry:
        val = entry[attr]
    # Try nested match
    elif "." in attr:
        sec, k = attr.split(".", 1)
        val = entry.get(sec, {}).get(k, "Unknown")
    
    # Fuzzy Fallback
    if val == "Unknown":
        if "loc" in attr: val = entry["location"]
        elif "elev" in attr: val = entry["elevation"]
        elif "built" in attr: val = entry["history"]["built_by"]

    if isinstance(val, list):
        return f"The {attr} of {entity} include: " + ", ".join(val)
        
    return f"FACT: The {attr} of {entity} is: {val}"


#  RUNTIME LOOP
if __name__ == "__main__":
    tokenizer, model, db = load_resources()
    print("-" * 50)
    print(" FORT GUIDE BOT (Final Polish)")
    print("-" * 50)
    
    while True:
        q = input("\nYou: ")
        if q.lower() in ["exit", "quit"]: break
        print(f"Bot: {get_bot_response(tokenizer, model, db, q)}")