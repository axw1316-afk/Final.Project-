import os
import csv
from datetime import datetime

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = CNNModel()
model.load_state_dict(torch.load("fashion_cnn.pth", map_location="cpu"))
model.eval()



class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
   
])



@torch.no_grad()
def predict_image(img_pil):
    img = transform(img_pil).unsqueeze(0)  
    outputs = model(img)
    probs = torch.softmax(outputs, dim=1)[0]
    pred_index = torch.argmax(probs).item()
    pred_name = class_names[pred_index]
    return pred_name, probs



CATALOG_DIR = "catalog"
METADATA_FILE = os.path.join(CATALOG_DIR, "metadata.csv")
os.makedirs(CATALOG_DIR, exist_ok=True)

def save_image_and_metadata(img_pil, pred_label):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{pred_label}_{timestamp}.png"

    label_dir = os.path.join(CATALOG_DIR, pred_label.replace("/", "_"))
    os.makedirs(label_dir, exist_ok=True)

    img_path = os.path.join(label_dir, filename)
    img_pil.save(img_path)

    file_exists = os.path.isfile(METADATA_FILE)
    with open(METADATA_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["filename", "label", "timestamp", "folder_path"])
        writer.writerow([filename, pred_label, timestamp, label_dir])

    return img_path



st.title("Clothing Classifier")
st.write("Upload a clothing image to get a predicted label and save it to your catalog.")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded image", use_column_width=True)

    if st.button("Predict"):
        pred_label, probs = predict_image(img)

        st.success(f"Predicted label: **{pred_label}**")

        top_probs, top_idxs = torch.topk(probs, 3)
        st.write("Top predictions:")
        for p, idx in zip(top_probs, top_idxs):
            st.write(f"- {class_names[idx]}: {p.item()*100:.1f}%")

        saved_path = save_image_and_metadata(img, pred_label)
        st.info(f"Image saved to: `{saved_path}`")
        st.info(f"Metadata logged in: `{METADATA_FILE}`")
