import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from ultralytics import YOLO
import av
import cv2
import numpy as np
from collections import Counter
import threading

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="YOLOv8 Object Detection",
    page_icon="🔍",
    layout="wide"
)

# ── Title & description ───────────────────────────────────────────────────────
st.title("🔍 Real-Time Object Detection")
st.markdown("**Powered by YOLOv8 + OpenCV** — detects 80 COCO classes live from your webcam")
st.divider()

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    confidence = st.slider("Confidence Threshold", 0.1, 1.0, 0.4, 0.05)
    model_choice = st.selectbox("Model Size", ["yolov8n.pt (Fastest)", "yolov8s.pt (Balanced)", "yolov8m.pt (Accurate)"])
    model_name = model_choice.split(" ")[0]

    st.divider()
    st.markdown("### 📋 Detectable Objects")
    st.markdown("""
    **People & Vehicles**  
    person, bicycle, car, motorcycle, bus, truck, boat

    **Electronics**  
    📱 cell phone, laptop, TV, keyboard, mouse, remote

    **Animals**  
    cat, dog, bird, horse, cow, elephant, bear

    **Household**  
    chair, sofa, bed, bottle, cup, book, clock
    """)

# ── Load model (cached) ───────────────────────────────────────────────────────
@st.cache_resource
def load_model(name):
    return YOLO(name)

model = load_model(model_name)

# ── Video processor ───────────────────────────────────────────────────────────
class YOLOProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model
        self.confidence = confidence
        self.lock = threading.Lock()
        self.detections = []

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Run inference
        results = self.model(img, conf=self.confidence, verbose=False)[0]
        annotated = results.plot()

        # Store detections for sidebar
        labels = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = self.model.names[cls_id]
            conf_val = float(box.conf[0])
            labels.append((label, conf_val))

        with self.lock:
            self.detections = labels

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

# ── Layout ────────────────────────────────────────────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📷 Live Camera Feed")
    ctx = webrtc_streamer(
        key="yolo-detection",
        video_processor_factory=YOLOProcessor,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col2:
    st.subheader("📊 Live Detections")
    detection_placeholder = st.empty()
    chart_placeholder = st.empty()

    if ctx.state.playing and ctx.video_processor:
        import time
        while True:
            with ctx.video_processor.lock:
                dets = list(ctx.video_processor.detections)

            if dets:
                counts = Counter([d[0] for d in dets])
                detection_text = ""
                for label, count in counts.most_common():
                    conf_avg = np.mean([d[1] for d in dets if d[0] == label]) * 100
                    detection_text += f"**{label}** × {count}  —  `{conf_avg:.0f}%`\n\n"
                detection_placeholder.markdown(detection_text)
                chart_placeholder.bar_chart({k: v for k, v in counts.items()})
            else:
                detection_placeholder.info("👀 No objects detected yet...")

            time.sleep(0.5)
    else:
        detection_placeholder.info("▶️ Click **START** above to begin detection")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "**AML Project** | YOLOv8 (Ultralytics) + OpenCV + Streamlit | "
    "Trained on COCO dataset — 80 object classes"
)
