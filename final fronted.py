import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
import numpy as np
import random
import os

# Load YOLO model
model = YOLO("yolov8n.pt", "v8")

# Load class names
with open(r"D:\cv2 project\yolo\coco.txt", "r") as f:
    class_list = f.read().strip().split("\n")

# Generate random colors for bounding boxes
detection_colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in class_list]

# Function to process images
def process_image(image_path):
    image = cv2.imread(image_path)
    detections = model.predict(source=[image], conf=0.45, save=False)

    if len(detections[0]) > 0:
        for box in detections[0].boxes:
            clsID = int(box.cls.numpy()[0])
            conf = box.conf.numpy()[0]
            bb = box.xyxy.numpy()[0]

            # Draw bounding box
            cv2.rectangle(
                image,
                (int(bb[0]), int(bb[1])),
                (int(bb[2]), int(bb[3])),
                detection_colors[clsID],
                2,
            )

            # Add label and confidence
            label = f"{class_list[clsID]} {conf:.2f}"
            cv2.putText(
                image,
                label,
                (int(bb[0]), int(bb[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )
    return image

# Function to process videos
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    output_frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform detection
        detections = model.predict(source=[frame], conf=0.45, save=False)
        if len(detections[0]) > 0:
            for box in detections[0].boxes:
                clsID = int(box.cls.numpy()[0])
                conf = box.conf.numpy()[0]
                bb = box.xyxy.numpy()[0]

                # Draw bounding box
                cv2.rectangle(
                    frame,
                    (int(bb[0]), int(bb[1])),
                    (int(bb[2]), int(bb[3])),
                    detection_colors[clsID],
                    2,
                )

                # Add label and confidence
                label = f"{class_list[clsID]} {conf:.2f}"
                cv2.putText(
                    frame,
                    label,
                    (int(bb[0]), int(bb[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )

        output_frames.append(frame)

    cap.release()
    return output_frames

def save_video(frames, output_path, fps):
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame)
    out.release()

# Streamlit interface
st.title("YOLO Object Detection App")
st.sidebar.header("Upload Media")

uploaded_file = st.sidebar.file_uploader("Choose an image or video file", type=["jpg", "png", "mp4", "avi", "mov"])

if uploaded_file is not None:
    # Save uploaded file to a temporary location
    temp_file_path = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[-1]).name
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(uploaded_file.read())

    file_extension = os.path.splitext(uploaded_file.name)[-1].lower()

    if st.button("Predict"):
        if file_extension in [".jpg", ".png"]:
            # Process image
            st.text("Processing image...")
            processed_image = process_image(temp_file_path)
            st.success("Processing complete!")

            # Convert image to RGB for displaying in Streamlit
            processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
            st.image(processed_image_rgb, caption="Processed Image", use_column_width=True)

            # Option to download the processed image
            result_path = "processed_image.jpg"
            cv2.imwrite(result_path, processed_image)
            with open(result_path, "rb") as file:
                st.download_button(
                    label="Download Processed Image",
                    data=file,
                    file_name="processed_image.jpg",
                    mime="image/jpeg",
                )

        elif file_extension in [".mp4", ".avi", ".mov"]:
            # Process video
            st.text("Processing video...")
            frames = process_video(temp_file_path)
            output_video_path = "output_video.mp4"
            save_video(frames, output_video_path, fps=30)

            st.success("Processing complete!")
            st.video(output_video_path)

            # Option to download the processed video
            with open(output_video_path, "rb") as video_file:
                st.download_button(
                    label="Download Processed Video",
                    data=video_file,
                    file_name="processed_video.mp4",
                    mime="video/mp4",
                )
