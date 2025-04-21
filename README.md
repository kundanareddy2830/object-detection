# object-detection
**🎯 YOLOv8n Real-Time Object Detection App**
A simple yet powerful real-time object detection web application built with Streamlit and YOLOv8n by Ultralytics. This app allows users to detect objects in real-time via webcam or through uploaded images/videos, with an intuitive UI and customizable detection settings.

🔍 Features

🚀 Real-time object detection using YOLOv8n

🎥 Support for Webcam or Image/Video uploads

🎯 Adjustable confidence threshold

🧠 Class filtering to display only selected object types

📸 Live FPS counter for performance insight

🛑 Stop button for on-demand termination

🧰 Tech Stack

Streamlit

OpenCV

Ultralytics YOLOv8

Pillow (PIL)

NumPy



> *Note:* Due to browser and server restrictions, *webcam access may not work* on cloud-hosted environments like Streamlit or mobile browsers.  
> Instead, you can upload images or videos to test the model.

---

## 🖥 Run Locally for Webcam Access

To enable webcam access (which doesn't work reliably in hosted environments), follow these steps:


### 1. Clone this repository
### 2.Create a virtual environment
### 3.Install Dependencies
### 4. Run the Streamlit App

🖼️ Usage

Select detection mode: Webcam or Upload Image/Video

Adjust confidence threshold and optionally filter classes

Click "Start Detection"

View annotated results with FPS indicator

Use the "🛑 Stop" button to terminate detection

📌 Notes

Webcam access is required for live detection mode

Larger models like yolov8s.pt, yolov8m.pt, or yolov8l.pt can also be used, but may affect performance

Use in a stable environment with camera permissions enabled


🙌 Acknowledgements


Ultralytics for YOLOv8

Streamlit for effortless UI creation

