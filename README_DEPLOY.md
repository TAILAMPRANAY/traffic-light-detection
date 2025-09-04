
# 🚦 Traffic Light Detection — Web UI (Streamlit)

A modern web interface for your YOLOv8-based traffic light detector.

## ▶️ Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## ☁️ Deploy (Streamlit Community Cloud)
1. Push this repo (at least `app.py` and `requirements.txt`) to GitHub.
2. Go to https://streamlit.io/cloud → “New app” → select your repo/branch.
3. App URL will look like: `https://<your-app-name>.streamlit.app`

## 🧠 Notes
- Works best with uploaded video files in the cloud.
- Uses the same HSV zone-based classification as your original script.
- If `yolov8n.pt` is not present, Ultralytics will download it on first run.
