"""AegisVision Streamlit Inspector Dashboard."""
import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config(page_title="AegisVision Inspector", layout="wide")

st.title("🛡️ AegisVision: Defect Inspection Assistant")
st.markdown("Upload an image to get an instant risk assessment and routing decision.")

# The API URL (will be the Hugging Face Space URL in production)
API_URL = "http://localhost:8000/predict"  # Change this after deployment

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Analyze Image"):
        with st.spinner("Running YOLO11s and Routing Policy..."):
            # Send to FastAPI backend
            files = {"file": uploaded_file.getvalue()}
            response = requests.post(API_URL, files=files)
            
            if response.status_code == 200:
                data = response.json()
                
                # Display routing decision
                if data["decision"] == "HUMAN_REVIEW":
                    st.error(f"🔴 **HUMAN REVIEW REQUIRED** (Risk Score: {data['danger_score']})")
                else:
                    st.success(f"🟢 **AUTO-ACCEPTED** (Risk Score: {data['danger_score']})")
                
                # Display details
                st.json(data)
            else:
                st.error("API request failed. Is the backend running?")
