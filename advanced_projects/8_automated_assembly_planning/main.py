import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5", trust_remote_code=True)

def parse_cad_metadata():
    # Simulated metadata extracted from a STEP file
    return "Assembly consists of BasePlate, Bearing, Shaft, and RetainingRing. The Bearing must sit in the BasePlate before the Shaft is inserted."

def generate_sequence():
    cad_data = parse_cad_metadata()
    prompt = f"Given the CAD metadata: '{cad_data}'. Generate a step-by-step robotic assembly sequence:"

    inputs = tokenizer(prompt, return_tensors="pt")

    print("Generating assembly sequence using SLM...")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=100)

    sequence = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Generated Sequence:")
    print(sequence)

if __name__ == "__main__":
    print("Starting Assembly Sequence Planning...")
    generate_sequence()
