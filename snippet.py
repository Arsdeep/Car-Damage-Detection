import torch
from PIL import Image
import argparse
import os

def process_image(image_path, output_dir="output"):
    # Load the YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.eval()

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

    print(f"Processed image saved at: {output_path}")


path = "image0.jpg"

process_image(path, f"{path}_output.png",)