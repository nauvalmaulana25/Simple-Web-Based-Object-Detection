import av
import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from ultralytics import YOLO

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(page_title="Simple Web Based Object Detection", layout="centered")

st.title("👁️ Simple Object Detection")
st.caption("Deteksi objek menggunakan kamera Laptop atau HP via Browser")

# --- Load Model ---
# Menggunakan @st.cache_resource agar model tidak di-load ulang setiap ada interaksi
@st.cache_resource
def load_model():
    # Pastikan file yolov10n.pt ada, atau ultralytics akan mencoba mendownloadnya
    # Jika yolov10n belum stabil di library standar, bisa ganti ke 'yolov8n.pt' sebagai fallback
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
### Cara Menggunakan:
1. Klik tombol **START** di atas.
2. Izinkan browser mengakses kamera.
3. **Untuk Laptop:** Akan langsung menggunakan webcam default.
4. **Untuk HP:**
   - Pastikan Laptop dan HP terhubung ke **Wi-Fi yang sama**.
   - Buka Command Prompt di laptop, ketik `ipconfig` (Windows) atau `ifconfig` (Mac/Linux) untuk melihat **IPv4 Address** laptop (misal: `192.168.1.5`).
   - Jalankan streamlit dengan perintah: `streamlit run app.py --server.address 0.0.0.0`
   - Buka browser di HP (Chrome/Safari), ketik alamat: `http://192.168.1.5:8501`.
   - Klik Start di HP, dan pilih kamera (biasanya browser HP akan meminta izin akses kamera).
""")