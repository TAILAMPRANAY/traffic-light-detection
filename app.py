
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
import time
import os

st.set_page_config(
    page_title="Traffic Light Detection",
    page_icon="🚦",
    layout="wide",
)

# ---------- Sidebar Controls ----------
st.sidebar.title("⚙️ Controls")
confidence_thresh = st.sidebar.slider("Detection Confidence", 0.1, 0.9, 0.3, 0.05)
max_frames = st.sidebar.number_input("Max Frames to Process (0 = all)", min_value=0, max_value=100000, value=0, step=50)
save_output = st.sidebar.checkbox("Save annotated video", value=False)
display_size = st.sidebar.select_slider("Display Size", options=["S", "M", "L"], value="M")

st.title("🚦 Traffic Light Detection System")
st.caption("YOLOv8 + HSV color segmentation • Modern web UI")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("🎥 Input")
    source = st.radio("Choose Source", ["Upload video"], captions=["Recommended"])
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])

    st.markdown("---")
    st.subheader("ℹ️ About")
    st.write(
        "This app detects **traffic lights** using **YOLOv8** and classifies their state "
        "(**Red / Yellow / Green**) via **HSV zone-based** segmentation."
    )

with col2:
    st.subheader("📺 Output")
    frame_placeholder = st.empty()
    stats_placeholder = st.empty()
    progress_bar = st.progress(0, text="Idle")

# Load model once and cache
@st.cache_resource(show_spinner=True)
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# HSV ranges (same logic as original script)
red_lower1 = np.array([0, 100, 100])
red_upper1 = np.array([10, 255, 255])
red_lower2 = np.array([160, 100, 100])
red_upper2 = np.array([180, 255, 255])
yellow_lower = np.array([20, 100, 100])
yellow_upper = np.array([35, 255, 255])
green_lower = np.array([40, 100, 100])
green_upper = np.array([85, 255, 255])

def dominant_color(region):
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    red_mask = cv2.inRange(hsv, red_lower1, red_upper1) | cv2.inRange(hsv, red_lower2, red_upper2)
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    green_mask = cv2.inRange(hsv, green_lower, green_upper)
    return {
        "Red": int(cv2.countNonZero(red_mask)),
        "Yellow": int(cv2.countNonZero(yellow_mask)),
        "Green": int(cv2.countNonZero(green_mask))
    }

def detect_color(crop):
    if crop.size == 0:
        return "Unknown"

    crop = cv2.resize(crop, (50, 150))
    h = crop.shape[0]

    top = crop[0:h // 3, :]
    middle = crop[h // 3:2 * h // 3, :]
    bottom = crop[2 * h // 3:, :]

    top_counts = dominant_color(top)
    mid_counts = dominant_color(middle)
    bot_counts = dominant_color(bottom)

    if top_counts["Red"] > 50:
        return "Red"
    elif mid_counts["Yellow"] > 50:
        return "Yellow"
    elif bot_counts["Green"] > 50:
        return "Green"
    else:
        return "Unknown"

def color_for_state(state):
    if state == "Red":
        return (0, 0, 255)
    if state == "Yellow":
        return (0, 255, 255)
    if state == "Green":
        return (0, 255, 0)
    return (128, 128, 128)

# Size mapping for display
size_map = {"S": 480, "M": 720, "L": 960}

def process_video(temp_video_path):
    cap = cv2.VideoCapture(temp_video_path)
    if not cap.isOpened():
        st.error("Could not open the uploaded video.")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    # Prepare writer if saving
    output_path = None
    if save_output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        output_path = os.path.join(tempfile.gettempdir(), "annotated_output.mp4")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    else:
        writer = None

    processed = 0
    start_time = time.time()

    tgt_h = size_map.get(display_size, 720)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        results = model(frame, verbose=False, conf=confidence_thresh)

        display_frame = frame.copy()

        # Iterate YOLO detections
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    # COCO class 9: traffic light
                    if int(box.cls) == 9:
                        conf = float(box.conf[0])
                        if conf < confidence_thresh:
                            continue
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(display_frame.shape[1]-1, x2), min(display_frame.shape[0]-1, y2)

                        roi = frame[y1:y2, x1:x2]
                        state = detect_color(roi)
                        color = color_for_state(state)

                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(display_frame, f"{state} ({conf:.2f})", (x1, max(0, y1 - 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Resize for UI
        scale = tgt_h / display_frame.shape[0]
        disp = cv2.resize(display_frame, (int(display_frame.shape[1] * scale), tgt_h))

        # Convert to RGB for Streamlit
        disp_rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(disp_rgb, caption=f"Frame {frame_idx}", use_column_width=True)

        processed += 1
        if total_frames > 0:
            progress_bar.progress(min(int(100 * frame_idx / total_frames), 100), text=f"Processing {frame_idx}/{total_frames}")
        else:
            progress_bar.progress(100, text=f"Processing frame {frame_idx}")

        # Write original-size annotated frame if saving
        if writer is not None:
            writer.write(display_frame)

        # Stop if reached max frames
        if max_frames and processed >= max_frames:
            break

    cap.release()
    if writer is not None:
        writer.release()

    end_time = time.time()
    elapsed = max(end_time - start_time, 1e-6)
    fps_runtime = processed / elapsed

    stats_placeholder.info(f"Processed **{processed}** frames in **{elapsed:.1f}s**  •  ~**{fps_runtime:.1f} FPS**")
    progress_bar.empty()

    return output_path

# -------- Main Run Button --------
if uploaded_file is not None:
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    with st.spinner("Processing video..."):
        out_path = process_video(temp_path)

    if out_path and os.path.exists(out_path):
        st.success("Annotated video ready for download.")
        with open(out_path, "rb") as f:
            st.download_button("⬇️ Download annotated video", f, file_name="annotated_output.mp4", mime="video/mp4")
else:
    st.info("Upload a video to begin.")
