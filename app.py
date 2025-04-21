


# import streamlit as st
# import cv2
# import time
# from ultralytics import YOLO
# from PIL import Image
# import io
# import numpy as np

# # App setup
# st.set_page_config(page_title="YOLOv8n Real-Time Detection", layout="centered")
# st.title("🎯 Real-Time Object Detection ")

# # Load YOLOv8n model
# model = YOLO("yolov8n.pt")

# # Sidebar for options
# st.sidebar.header("🔧 Detection Settings")
# confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)
# selected_classes = st.sidebar.multiselect("Show only classes:", model.model.names.values())

# # Sidebar options for webcam or file upload
# detection_mode = st.sidebar.radio("Select Detection Mode", ("Webcam", "Upload Image/Video"))

# # Start detection
# start = st.button("Start Detection")
# stop_placeholder = st.empty()
# status = st.empty()

# def detect_webcam():
#     cap = cv2.VideoCapture(0)
#     stop_btn = stop_placeholder.button("🛑 Stop")
#     stframe = st.empty()

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         start_time = time.time()
#         results = model(frame, conf=confidence)

#         # Apply class filter
#         if selected_classes:
#             results[0].boxes = results[0].boxes[[model.model.names[int(cls)] in selected_classes for cls in results[0].boxes.cls]]

#         annotated = results[0].plot()
#         annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

#         # FPS calculation
#         end_time = time.time()
#         fps = 1 / (end_time - start_time + 1e-8)

#         stframe.image(annotated, channels="RGB", use_container_width=True)
#         status.info(f"📸 FPS: {fps:.2f} | Showing: {', '.join(selected_classes) if selected_classes else 'All'}")

#         if stop_btn:
#             break

#     cap.release()
#     status.success("✅ Detection stopped.")

# def detect_from_file(uploaded_file):
#     # Load the uploaded file (image or video)
#     if uploaded_file.type.startswith("image"):
#         image = Image.open(uploaded_file)
#         image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
#         process_image(image)
#     elif uploaded_file.type.startswith("video"):
#         cap = cv2.VideoCapture(uploaded_file)
#         stop_btn = stop_placeholder.button("🛑 Stop")
#         stframe = st.empty()

#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             process_image(frame)

#             if stop_btn:
#                 break

#         cap.release()
#         status.success("✅ Detection stopped.")

# def process_image(frame):
#     start_time = time.time()
#     results = model(frame, conf=confidence)

#     # Apply class filter
#     if selected_classes:
#         results[0].boxes = results[0].boxes[[model.model.names[int(cls)] in selected_classes for cls in results[0].boxes.cls]]

#     annotated = results[0].plot()
#     annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

#     # FPS calculation
#     end_time = time.time()
#     fps = 1 / (end_time - start_time + 1e-8)

#     st.image(annotated, channels="RGB", use_container_width=True)
#     status.info(f"📸 FPS: {fps:.2f} | Showing: {', '.join(selected_classes) if selected_classes else 'All'}")

# # Handle the detection based on mode selected
# if detection_mode == "Webcam":
#     if start:
#         detect_webcam()
# elif detection_mode == "Upload Image/Video":
#     uploaded_file = st.file_uploader("Upload Image/Video", type=["jpg", "png", "jpeg", "mp4", "avi"])
#     if uploaded_file is not None:
#         if start:
#             detect_from_file(uploaded_file)
import streamlit as st
import cv2
import time
from ultralytics import YOLO
from PIL import Image
import numpy as np

# App setup
st.set_page_config(page_title="YOLOv8n Real-Time Detection", layout="centered")
st.title("🎯 Real-Time Object Detection")

# Load model
model = YOLO("yolov8n.pt")

# Sidebar settings
st.sidebar.header("🔧 Detection Settings")
confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)
selected_classes = st.sidebar.multiselect("Show only classes:", model.model.names.values())
detection_mode = st.sidebar.radio("Select Detection Mode", ("Webcam", "Upload Image/Video"))

# Placeholders
stop_button = st.empty()
status = st.empty()

# Function to process image/frame
def process_frame(frame):
    start_time = time.time()
    results = model(frame, conf=confidence)

    # Filter classes
    if selected_classes:
        results[0].boxes = results[0].boxes[
            [model.model.names[int(cls)] in selected_classes for cls in results[0].boxes.cls]
        ]

    annotated = results[0].plot()
    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    end_time = time.time()
    fps = 1 / (end_time - start_time + 1e-8)

    return annotated, fps

# Webcam detection
def detect_webcam():
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        st.error("Unable to access webcam.")
        return

    stframe = st.empty()
    stop = stop_button.button("🛑 Stop Detection")

    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to grab frame.")
            break

        annotated, fps = process_frame(frame)
        stframe.image(annotated, channels="RGB", use_container_width=True)
        status.info(f"📸 FPS: {fps:.2f} | Showing: {', '.join(selected_classes) if selected_classes else 'All'}")

        # Continuously check the stop button
        if stop_button.button("🛑 Stop Detection"):
            break

    cap.release()
    status.success("✅ Detection stopped.")

# Image/Video file detection
def detect_file(uploaded_file):
    if uploaded_file.type.startswith("image"):
        image = Image.open(uploaded_file)
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        annotated, fps = process_frame(frame)
        st.image(annotated, channels="RGB", use_container_width=True)
        status.info(f"📸 FPS: {fps:.2f} | Showing: {', '.join(selected_classes) if selected_classes else 'All'}")

    elif uploaded_file.type.startswith("video"):
        cap = cv2.VideoCapture(uploaded_file.name)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            annotated, fps = process_frame(frame)
            stframe.image(annotated, channels="RGB", use_container_width=True)
            status.info(f"📸 FPS: {fps:.2f} | Showing: {', '.join(selected_classes) if selected_classes else 'All'}")

            if stop_button.button("🛑 Stop Detection"):
                break

        cap.release()
        status.success("✅ Detection stopped.")

# Main control
# if detection_mode == "Webcam":
#     if st.button("Start Detection"):
#         detect_webcam()

if detection_mode == "Webcam":
    st.info("Note: Webcam access only works when this app is run on your *local machine*.")
    st.warning("Due to browser and server restrictions, webcam won’t work in deployed versions (like Streamlit Cloud).")

    st.markdown("""
        - *To use webcam detection*, please run this app locally:
        bash
        git clone https://github.com/yourusername/your-repo.git
        cd your-repo
        pip install -r requirements.txt
        streamlit run app.py
        
        - Or try uploading an image or video below.
    """)

    if st.button("Start Detection (Local Only)"):
        detect_webcam()

elif detection_mode == "Upload Image/Video":
    uploaded_file = st.file_uploader("Upload Image/Video", type=["jpg", "jpeg", "png", "mp4", "avi"])
    if uploaded_file is not None:
        if st.button("Start Detection"):
            detect_file(uploaded_file)
