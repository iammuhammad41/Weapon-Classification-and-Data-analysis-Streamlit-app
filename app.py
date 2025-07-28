import os
import streamlit as st
import torch
import numpy as np
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from train import train_ds, val_ds  # reuse train.py's datasets

# --- Config ---
MODEL_PATH = "models/weapon_resnet18.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load Model ---
@st.cache(allow_output_mutation=True)
def load_model():
    model = models.resnet18(pretrained=False)
    in_feats = model.fc.in_features
    model.fc = torch.nn.Linear(in_feats, len(train_ds.classes))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model

model = load_model()
classes = train_ds.classes

# --- Image Transform ---
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("", ["Home", "EDA", "Classify"])

# --- Home ---
if page == "Home":
    st.title("🔫 Weapon Classification & Data Analysis")
    st.markdown("""
    This app trains a ResNet18 model to classify weapon images into:
    **`{}`**  
    Upload your own image under **Classify** to see predictions.
    """.format(", ".join(classes)))
    st.image("training_metrics.png", use_column_width=True)
    st.write("Model training metrics (accuracy & loss).")

# --- EDA ---
elif page == "EDA":
    st.title("📊 Exploratory Data Analysis")
    st.subheader("Class Distribution")
    counts = {cls: len(os.listdir(os.path.join("data/train", cls))) for cls in classes}
    fig, ax = plt.subplots()
    sns.barplot(x=list(counts.keys()), y=list(counts.values()), ax=ax)
    ax.set_ylabel("Number of Images")
    st.pyplot(fig)

    st.subheader("Sample Images")
    cols = st.columns(len(classes))
    for idx, cls in enumerate(classes):
        img_list = os.listdir(os.path.join("data/train", cls))[:3]
        with cols[idx]:
            st.markdown(f"**{cls}**")
            for img_name in img_list:
                img = Image.open(os.path.join("data/train", cls, img_name))
                st.image(img, width=100)

# --- Classify ---
elif page == "Classify":
    st.title("🖼️ Upload & Classify")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg","png","jpeg"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)
        input_tensor = transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)
            top5 = torch.topk(probs, k=3)
        st.write("### Predictions:")
        for prob, idx in zip(top5.values.cpu(), top5.indices.cpu()):
            st.write(f"{classes[idx]}: {prob*100:.2f}%")
