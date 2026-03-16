import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd
from model import get_model
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import cv2
import numpy as np
def reshape_transform(tensor, height=14, width=14):
    tensor = tensor[:, 1:, :]  
    result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.permute(0, 3, 1, 2)
    return result


st.set_page_config(
    page_title="Human Disease Detector",
    page_icon="",
    layout="centered"
)


st.markdown("""
    <style>
    .main {
        background-color: #f4f8f4;
    }
    .stButton>button {
        background-color: #2e7d32;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 16px;
    }
    .result-box {
        padding: 20px;
        border-radius: 15px;
        background-color: #ffffff;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)


st.markdown("<h1 style='text-align: center;'>🧬 Human Disease Detection System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>AI-powered ConViT Model for Medical Image Classification</p>", unsafe_allow_html=True)
st.divider()

CLASS_NAMES = ["Benign", "Malignant", "Normal"]

NUM_CLASSES = len(CLASS_NAMES)


@st.cache_resource
def load_model():
    model = get_model(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load("best_model.pth", map_location="cpu",weights_only=True))
    model.eval()
    return model

model = load_model()
target_layers = [model.blocks[-1].norm1]
cam = GradCAM(
    model=model,
    target_layers=target_layers,
    reshape_transform=reshape_transform
)

def transform_image(image):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform(image).unsqueeze(0)


uploaded_file = st.file_uploader(
    "📷 Upload Masked disease image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    col1, col2 = st.columns([1, 1])

    image = Image.open(uploaded_file).convert("RGB")

    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)

    input_tensor = transform_image(image)
    model.zero_grad()

    outputs = model(input_tensor)
    probabilities = F.softmax(outputs, dim=1)[0]

    confidence, predicted_class = torch.max(probabilities, 0)

    targets = [ClassifierOutputTarget(predicted_class.item())]

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0]
    predicted_label = CLASS_NAMES[predicted_class.item()]
    confidence_score = confidence.item() * 100
   
    rgb_img = np.array(image.resize((224, 224))).astype(np.float32) / 255.0

    visualization = show_cam_on_image(
        rgb_img,
        grayscale_cam,
        use_rgb=True
    )

    
    with col2:
        st.markdown("###  Prediction Result")
        
        st.markdown(f"""
            <div class="result-box">
                <h2 style='color:#2e7d32;'>{predicted_label}</h2>
                <h4>Confidence: {confidence_score:.2f}%</h4>
            </div>
        """, unsafe_allow_html=True)

        st.progress(int(confidence_score))

    st.divider()
    st.subheader("🔍 Model Attention (Grad-CAM)")

    st.image(
        visualization,
        caption="Highlighted regions the model focused on",
        use_column_width=True
    )

   
    st.subheader("📊 Top 3 Predictions")

    probs_df = pd.DataFrame({
        "Disease": CLASS_NAMES,
        "Probability (%)": probabilities.detach().cpu().numpy() * 100
    })

    probs_df = probs_df.sort_values(by="Probability (%)", ascending=False)

    st.bar_chart(probs_df.set_index("Disease"))

    st.dataframe(probs_df.head(3), use_container_width=True)

    
    st.subheader("🩺 Suggested Action")

    if predicted_label == "Normal":
        st.success("No abnormality detected. The sample appears normal.")

    elif predicted_label == "Benign":
        st.warning("Benign condition detected. It is generally non-cancerous, but medical consultation is recommended for confirmation.")

    elif predicted_label == "Malignant":
        st.error("Malignant condition detected. Immediate medical consultation with a specialist is strongly recommended.")

