# 🔍 Real-Time Object Detection — AML Project

A real-time object detection application built with **YOLOv8**, **OpenCV**, and **Streamlit**.

## 🚀 Features
- Real-time webcam object detection using YOLOv8
- Streamlit web app with live video feed
- Detects **80 COCO classes** including 📱 cell phone, 💻 laptop, 👤 person, 🪑 chair, 🍶 bottle and more
- Confidence threshold control via sidebar slider
- Live detection chart and labels

## 📁 Project Structure
```
├── streamlit_app.py              # Streamlit web app (YOLOv8)
├── yolo_detection.py             # Standalone YOLOv8 script
├── real_time_object_detection.py # Original MobileNet SSD script
├── MobileNetSSD_deploy.prototxt.txt
└── MobileNetSSD_deploy.caffemodel
```

## ⚙️ Installation
```bash
pip install opencv-python imutils ultralytics streamlit streamlit-webrtc av
```

## ▶️ Run

### Streamlit App (Recommended)
```bash
streamlit run streamlit_app.py
```
Open http://localhost:8501 in your browser, click START, allow camera access.

### Standalone Script
```bash
python yolo_detection.py
```
Press `q` to quit.

## 🧠 Models
| Model | Classes | Speed |
|-------|---------|-------|
| MobileNet SSD | 21 (PASCAL VOC) | Fast |
| YOLOv8n | 80 (COCO) | Fast |

## 📦 Tech Stack
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [OpenCV](https://opencv.org/)
- [Streamlit](https://streamlit.io/)
- [streamlit-webrtc](https://github.com/whitphx/streamlit-webrtc)
