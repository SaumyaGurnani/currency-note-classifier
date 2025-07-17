import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models

import os
import gdown

model_path = "currency_classifier.pth"
file_id = "1Y8B66PtBs3E27Y4SfjXe6Zz27nC-zUeL"
gdown_url = f"https://drive.google.com/uc?id={file_id}"

# 🔽 Download model from Google Drive if not present
if not os.path.exists(model_path):
    with st.spinner("Downloading model weights..."):
        gdown.download(gdown_url, model_path, quiet=False)

# ---- Load Model ----
class_names = ['10', '100', '20', '200', '2000', '50', '500']  # Update as per your dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18()
num_ftrs = model.fc.in_features

model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 512),
    nn.ReLU(),
    nn.Dropout(0.25),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, len(class_names))
)

model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# ---- Transforms ----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ---- Streamlit UI ----
st.title("Indian Currency Note Classifier")
st.write("Upload an image of a currency note and let the model predict the denomination.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_container_width=True)

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, preds = torch.max(outputs, 1)
        predicted_label = class_names[preds.item()]
    
    st.success(f"Predicted Currency: ₹{predicted_label}")
