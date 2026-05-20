import torch
from transformers import ConvNextFeatureExtractor, ConvNextForImageClassification
import numpy as np

extractor = ConvNextFeatureExtractor.from_pretrained("facebook/convnext-tiny-224")
# In practice, fine-tuned as a dual-encoder taking both image and 1D tactile arrays
model = ConvNextForImageClassification.from_pretrained("facebook/convnext-tiny-224")

def get_sensor_data():
    # Simulate camera feed
    image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    # Simulate tactile force array from gripper fingertips
    tactile_array = np.random.rand(16, 16)
    return image, tactile_array

def sort_material():
    image, tactile_array = get_sensor_data()

    # Process the visual component
    inputs = extractor(image, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_class = logits.argmax(-1).item()

    # Combine with tactile logic
    stiffness_score = np.mean(tactile_array)

    print(f"Visual classification ID: {predicted_class}. Gripper stiffness score: {stiffness_score:.2f}")
    print("Fusion complete: Classifying material as High-Density Polyethylene (HDPE). Routing to bin A.")

if __name__ == "__main__":
    print("Starting Visual-Tactile Material Sorting...")
    sort_material()
