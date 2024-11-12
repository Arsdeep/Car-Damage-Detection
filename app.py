import streamlit as st
import torch
from PIL import Image
import os
import tempfile
import shutil
import logging
from pathlib import Path
from typing import Optional, Tuple
import yaml
from dataclasses import dataclass
from contextlib import contextmanager
import time
import io

# Set page configuration to wide mode
st.set_page_config(layout="wide", page_title="Car Damage Detection", page_icon="🚗")

# Configuration class
@dataclass
class Config:
    model_name: str = 'yolov5s'
    model_source: str = 'ultralytics/yolov5'
    confidence_threshold: float = 0.25
    allowed_extensions: tuple = ('jpg', 'jpeg', 'png')
    max_image_size: int = 2560  # Increased maximum dimension
    output_quality: int = 95
    use_cuda: bool = torch.cuda.is_available()
    display_width: int = 1000  # Default display width for output
    min_confidence: float = 0.25  # Minimum confidence threshold

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def apply_custom_css():
    """Apply custom CSS to make the interface larger and more modern."""
    st.markdown("""
        <style>
        .main > div {
            padding-top: 2rem;
        }
        .stButton>button {
            height: 3rem;
            font-size: 1.2rem;
        }
        .uploadedFile {
            margin: 2rem 0;
        }
        .stAlert > div {
            padding: 1rem;
            font-size: 1.1rem;
        }
        .stImage > img {
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_yolo_model(model_source: str, model_name: str, confidence_threshold: float):
    """Load the YOLOv5 model with caching."""
    try:
        model = torch.hub.load(model_source, model_name, pretrained=True)
        model.eval()
        model.conf = confidence_threshold
        if torch.cuda.is_available():
            model.cuda()
        logger.info(f"Model loaded successfully on {'cuda' if torch.cuda.is_available() else 'cpu'}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

class ImageProcessor:
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        
    def ensure_model_loaded(self) -> None:
        """Ensure the model is loaded using the cached function."""
        if self.model is None:
            self.model = load_yolo_model(
                self.config.model_source,
                self.config.model_name,
                self.config.confidence_threshold
            )

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess the image to ensure it meets requirements."""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        if max(image.size) > self.config.max_image_size:
            ratio = self.config.max_image_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.LANCZOS)
        
        return image

    def process_image(self, image: Image.Image) -> Tuple[Image.Image, dict]:
        """Process the image and return the output image and detection results."""
        self.ensure_model_loaded()

        try:
            processed_image = self.preprocess_image(image)
            
            start_time = time.time()
            with torch.cuda.amp.autocast() if self.config.use_cuda else contextmanager(lambda: iter([None]))():
                results = self.model(processed_image)
            inference_time = time.time() - start_time
            
            # Filter detections based on confidence threshold
            results.xyxy[0] = results.xyxy[0][results.xyxy[0][:, 4] >= self.config.confidence_threshold]
            
            detections = results.pandas().xyxy[0].to_dict('records')
            
            # Render the results with custom colors and thicker lines
            for det in results.xyxy[0]:
                if det[4] >= self.config.confidence_threshold:
                    results.names[int(det[5])]  # class name
            
            results.render()
            output_image = Image.fromarray(results.ims[0])
            
            return output_image, {
                'inference_time': inference_time,
                'detections': detections
            }
                
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise

class DamageDetectionApp:
    def __init__(self):
        self.config = Config()
        self.processor = ImageProcessor(self.config)
        
    def setup_sidebar(self):
        """Setup the sidebar with enhanced configuration options."""
        st.sidebar.header("Detection Settings")
        
        # Model selection
        model_options = {
            'YOLOv5s': 'yolov5s',
            'YOLOv5m': 'yolov5m',
            'YOLOv5l': 'yolov5l',
            'YOLOv5x': 'yolov5x'
        }
        selected_model = st.sidebar.selectbox(
            "Select Model",
            options=list(model_options.keys()),
            index=0
        )
        self.config.model_name = model_options[selected_model]
        
        # Confidence threshold
        self.config.confidence_threshold = st.sidebar.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=self.config.min_confidence,
            step=0.05
        )
        
        # Display settings
        # st.sidebar.header("Display Settings")
        # self.config.display_width = st.sidebar.slider(
        #     "Display Width",
        #     min_value=500,
        #     max_value=2000,
        #     value=1000,
        #     step=100
        # )
        
        self.config.display_width = 700
        
        # GPU info
        st.sidebar.header("System Info")
        gpu_info = "GPU Available ✅" if torch.cuda.is_available() else "CPU Only ❌"
        st.sidebar.info(gpu_info)
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            st.sidebar.text(f"GPU: {gpu_name}")

    def display_results(self, output_image: Image.Image, results: dict):
        """Display the processed image and detection results with enhanced layout."""
        # Create two columns for layout
        col1, col2 = st.columns([0.7, 0.3])
        
        with col1:
            # Display the processed image
            st.image(output_image, caption="Processed Image with Detections", width=self.config.display_width)
        
        with col2:
            # Create a card-like container for results
            with st.container():
                st.markdown("### Detection Results")
                st.write(f"Processing time: {results['inference_time']:.2f} seconds")
                
                if results['detections']:
                    for det in results['detections']:
                        confidence = det['confidence'] * 100
                        # Create a more visually appealing detection display
                        st.markdown(f"""
                            <div style="
                                padding: 10px;
                                border-radius: 5px;
                                background-color: rgba(0, 100, 255, 0.1);
                                margin: 5px 0;
                            ">
                                <strong>{det['name']}</strong>: {confidence:.1f}%
                            </div>
                        """, unsafe_allow_html=True)
                else:
                    st.warning("No objects detected")

    def run(self):
        """Run the enhanced Streamlit application."""
        apply_custom_css()
        
        # Main header with custom styling
        st.markdown("""
            <h1 style='text-align: center; color: #1E88E5; padding: 1.5rem 0;'>
                🚗 Car Damage Detection
            </h1>
        """, unsafe_allow_html=True)
        
        self.setup_sidebar()
        
        # Create a centered upload button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            uploaded_file = st.file_uploader(
                "Upload an image",
                type=self.config.allowed_extensions,
                help="Supported formats: JPG, JPEG, PNG"
            )

        if uploaded_file is not None:
            try:
                # Show processing message
                with st.spinner("📸 Processing image..."):
                    # Load and process image
                    img = Image.open(uploaded_file)
                    output_image, results = self.processor.process_image(img)
                    
                    # Display results
                    self.display_results(output_image, results)
                    
            except Exception as e:
                st.error(f"❌ Error processing image: {str(e)}")
                logger.error(f"Application error: {str(e)}")
                if st.checkbox("Show error details"):
                    st.exception(e)

if __name__ == "__main__":
    app = DamageDetectionApp()
    app.run()