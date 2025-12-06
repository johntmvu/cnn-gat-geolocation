import gradio as gr
import torch
from torchvision import transforms
from PIL import Image
import pandas as pd
import torch.nn as nn
import torchvision.models as models

# Load the mapping CSV
mapping_df = pd.read_csv("data/gldv2_micro/train_label_to_category.csv")
id_to_location = dict(zip(mapping_df["landmark_id"], mapping_df["category"]))

# Load the trained model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=False)
checkpoint = torch.load("models/best_resnet50.pth", map_location=DEVICE)

# Dynamically adjust the fc layer to match the checkpoint
num_classes = checkpoint["fc.weight"].size(0)  # Get the number of classes from the checkpoint
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Load the checkpoint
model.load_state_dict(checkpoint)
model.eval().to(DEVICE)

# Define the image preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def map_prediction_to_location(predicted_landmark_id):
    # Map the predicted landmark_id to a location
    location = id_to_location.get(predicted_landmark_id, "Unknown")
    return location

# Update the predict_location function to include mapping
def predict_location(image):
    # Preprocess the image
    image = Image.fromarray(image).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Make a prediction
    with torch.no_grad():
        outputs = model(image.to(DEVICE))
        _, predicted = torch.max(outputs, 1)  # Get the predicted class index

    # Map the predicted class index to a location
    predicted_landmark_id = predicted.item()
    location = map_prediction_to_location(predicted_landmark_id)

    return location

# Create a Gradio interface
interface = gr.Interface(
    fn=predict_location,
    inputs=gr.Image(type="numpy", label="Upload an Image"),
    outputs=gr.Textbox(label="Predicted Location"),
    title="Landmark Location Predictor",
    description="Upload an image to predict its landmark location."
)

# Launch the interface
if __name__ == "__main__":
    interface.launch()