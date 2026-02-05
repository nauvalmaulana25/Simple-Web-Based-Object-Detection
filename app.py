import av
import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from ultralytics import YOLO

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(page_title="Simple Web Based Object Detection", layout="centered")

st.title("👁️ Simple Object Detection")
st.caption("Object Detection using Laptop or Phone camera via Browser")

# --- Load Model ---
# Menggunakan @st.cache_resource agar model tidak di-load ulang setiap ada interaksi
@st.cache_resource
def load_model():
    return YOLO(r"dailyobjects.pt") 

try:
    model = load_model()
except Exception as e:
    st.error(f"Gagal memuat model. Pastikan 'dailyobjects.pt' tersedia. Error: {e}")
    st.stop()

# --- Sidebar Pengaturan ---
st.sidebar.header("Pengaturan")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)

# --- Logic Pemrosesan Frame (Callback) ---
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")

    results = model(img, conf=conf_threshold)

    annotated_frame = results[0].plot()

    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# --- WebRTC Streamer ---
# Konfigurasi STUN server (opsional tapi disarankan untuk koneksi HP ke Laptop via WiFi)
rtc_configuration = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

st.write("### Kamera Feed")

# Komponen utama webrtc
webrtc_streamer(
    key="yolo-stream",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=rtc_configuration,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True,
)

st.markdown("""
---
### How to Use:
1. Click **START** Button on top.
2. Allow browser access the camera.
3. **For Laptop:** will use default webcam .
4. **For Phone:**
   - Choose back or front camera.
   - Happy Trying

""")
