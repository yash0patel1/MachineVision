
import streamlit as st
import cv2
import numpy as np
import torch
from PIL import Image
import os
import time
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support
from io import BytesIO

# Function to display images from a folder (inventory of images)
def load_images_from_directory(directory_path):
    """
    Loads all images from a specified directory.
    """
    image_files = [f for f in os.listdir(directory_path) if f.endswith(('png', 'jpg', 'jpeg'))]
    image_paths = [os.path.join(directory_path, f) for f in image_files]
    return image_paths

# Function for simulating image enhancement (placeholder for CycleGAN or other model)
def enhance_image(image):
    """
    Enhance the image's brightness and contrast as a placeholder for CycleGAN or other advanced models.
    This will be a simple enhancement for demonstration.
    """
    enhanced_image = cv2.convertScaleAbs(image, alpha=1.5, beta=20)  # Simple enhancement technique
    return enhanced_image

# Function to perform object detection using YOLOv5
def detect_objects(image):
    """
    Use YOLOv5 pre-trained model to detect objects in the enhanced image.
    """
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Load small YOLOv5 model
    results = model(image)  # Run object detection
    return results

# Function to calculate precision, recall, and F1-score for object detection performance
def calculate_metrics(results, true_labels):
    """
    Calculate precision, recall, and F1 score using the predicted and ground truth labels.
    """
    pred_labels = [result['name'] for result in results.names]  # Predicted labels from YOLOv5
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average='weighted')
    return precision, recall, f1

# Function to display additional information like description and instructions
def display_project_description():
    """
    Displays the project description and how to use the app.
    """
    st.markdown("""
        # Low-Light Image Enhancement and Object Detection

        This project demonstrates the combination of image enhancement and object detection to improve performance in low-light environments. The key steps include:

        1. **Image Selection**: Choose an image from the provided inventory.
        2. **Image Enhancement**: The chosen image will be enhanced to simulate low-light enhancement.
        3. **Object Detection**: After enhancement, objects like pedestrians, vehicles, and road signs are detected using a state-of-the-art YOLOv5 model.
        4. **Evaluation**: Precision, recall, and F1-score metrics are computed based on the detection results compared with ground truth labels.

        ## How to Use:
        - **Step 1**: Select an image from the sidebar.
        - **Step 2**: View the original and enhanced image.
        - **Step 3**: See the detection results and evaluation metrics.
    """)

# Function to display a loading spinner while performing time-consuming tasks
def show_loading_spinner():
    with st.spinner('Processing... Please wait'):
        time.sleep(2)  # Simulate time for processing (remove for real-time performance)
        st.success('Processing complete!')

# Main Streamlit Application
def main():
    """
    Main function for the Streamlit app.
    """
    st.set_page_config(page_title="Low-Light Enhancement and Object Detection", page_icon=":guardsman:", layout="wide")
    display_project_description()

    # Set directory for images (put your images in the folder you specify here)
    directory_path = 'your_image_directory'  # Update with the correct path to your image directory
    
    # Verify the directory exists
    if not os.path.exists(directory_path):
        st.error(f"The directory {directory_path} does not exist. Please update the path.")
        return
    
    # Load images from the directory
    image_paths = load_images_from_directory(directory_path)
    
    if len(image_paths) == 0:
        st.error("No images found in the directory. Please make sure the directory contains images.")
        return
    
    # Sidebar: Allow user to select an image
    st.sidebar.header("Select an Image")
    selected_image_path = st.sidebar.selectbox("Choose an image:", image_paths)
    
    # Load and display the selected image
    selected_image = Image.open(selected_image_path)
    st.image(selected_image, caption="Original Image", use_column_width=True)
    
    # Convert the selected image to OpenCV format for enhancement and detection
    image_cv = np.array(selected_image)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
    
    # Image Enhancement
    st.header("Image Enhancement")
    show_loading_spinner()  # Show loading spinner while enhancing image
    enhanced_image = enhance_image(image_cv)
    
    # Show enhanced image
    enhanced_image_pil = Image.fromarray(enhanced_image)
    st.image(enhanced_image_pil, caption="Enhanced Image", use_column_width=True)
    
    # Object Detection
    st.header("Object Detection Results")
    show_loading_spinner()  # Show loading spinner while detecting objects
    detection_results = detect_objects(enhanced_image)
    
    # Display Detected Objects with Bounding Boxes
    st.subheader("Detected Objects with Confidence Scores")
    detected_image = detection_results.render()[0]  # Add bounding boxes and labels
    st.image(detected_image, caption="Detection Results", use_column_width=True)
    
    # Extract detected object details and show in a table format
    detected_objects = detection_results.pandas().xywh  # Detected objects as a pandas dataframe
    st.write("Objects Detected:")
    st.dataframe(detected_objects)
    
    # Simulate Ground Truth Labels (for evaluation purposes)
    st.write("Evaluation Metrics (precision, recall, F1-score):")
    true_labels = ["car", "person", "person", "truck", "car"]  # Simulated ground truth labels

    # Calculate Precision, Recall, and F1-score
    precision, recall, f1 = calculate_metrics(detection_results, true_labels)
    
    # Display evaluation metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="Precision", value=f"{precision:.2f}")
    
    with col2:
        st.metric(label="Recall", value=f"{recall:.2f}")
    
    with col3:
        st.metric(label="F1-Score", value=f"{f1:.2f}")

# Run the Streamlit app
if __name__ == '__main__':
    main()
