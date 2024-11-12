import streamlit as st
import torch
from PIL import Image
import os

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.eval()
    
# Define the image processing function
def process_image(image_path, output_dir="output"):

    # Load the image
    img = Image.open(image_path)
    
    # Perform inference
    results = model(img)
    results.render()  # Updates results.ims with boxes and labels

    # Save the processed image
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    Image.fromarray(results.ims[0]).save(output_path)

    return output_path

# Streamlit app
st.title("Car Damage Detection App")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Save the uploaded file temporarily
    temp_image_path = f"temp_{uploaded_file.name}"
    with open(temp_image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Run object detection
    output_image_path = process_image(temp_image_path)
    
    # Display the results
    st.image(output_image_path, caption="Processed Image with Detections")
    
    # Clean up the temporary file
    os.remove(temp_image_path)
