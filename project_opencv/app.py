import streamlit as st
import cv2
import os
import time
import numpy as np
import pyttsx3
from collections import deque
from ultralytics import YOLO
import plotly.graph_objects as go

# ===== MediaPipe Tasks API (Python 3.12 SAFE) =====
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ================= BASIC SETUP =================
st.set_page_config(page_title="Person Detection – 3D Face", layout="wide")
st.title("Person Detection – Local Web App with 3D Face")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Ensure folders exist
os.makedirs(os.path.join(BASE_DIR, "captured/body"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "captured/face"), exist_ok=True)

# ================= SIDEBAR =================
st.sidebar.header("Controls")
run = st.sidebar.toggle("Start Detection", value=False)
sound_on = st.sidebar.toggle("Sound ON", value=True)
status_box = st.sidebar.empty()

# ================= UI LAYOUT =================
col1, col2, col3 = st.columns([3, 2, 2])

with col1:
    st.subheader("Live Camera")
    frame_placeholder = st.empty()

with col2:
    st.subheader("Last Captured")
    face_placeholder = st.empty()
    body_placeholder = st.empty()

with col3:
    st.subheader("3D Face View")
    face_3d_placeholder = st.empty()

# ================= MODELS =================
yolo = YOLO(os.path.join(BASE_DIR, "yolov8n.pt"))

face_net = cv2.dnn.readNetFromCaffe(
    os.path.join(BASE_DIR, "deploy.prototxt"),
    os.path.join(BASE_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
)

# ===== FaceLandmarker init =====
base_options = python.BaseOptions(
    model_asset_path=os.path.join(BASE_DIR, "face_landmarker.task")
)

face_options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)

face_landmarker = vision.FaceLandmarker.create_from_options(face_options)

orb = cv2.ORB_create(500)

# ================= VOICE =================
engine = pyttsx3.init()
engine.setProperty("rate", 160)

def speak_person_detected():
    if sound_on:
        engine.say("Person detected")
        engine.runAndWait()

# ================= PARAMETERS =================
FACE_BLUR_THRESHOLD = 100
MIN_BODY_W, MIN_BODY_H = 120, 220
FACE_MATCH_THRESHOLD = 0.65
COOLDOWN = 1.5

last_capture_time = 0
known_faces = []
frame_buffer = deque(maxlen=20)

# ================= HELPERS =================
def sharpness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def face_descriptor(face):
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    _, des = orb.detectAndCompute(gray, None)
    return des

def is_same_person(d1, d2):
    if d1 is None or d2 is None:
        return False
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(d1, d2)
    return len(matches) / max(len(d1), len(d2)) > FACE_MATCH_THRESHOLD if matches else False

def latest_image(folder):
    files = sorted(
        [os.path.join(folder, f) for f in os.listdir(folder)],
        key=os.path.getmtime,
        reverse=True
    )
    return files[0] if files else None

# ===== 3D FACE GENERATION (Tasks API) =====
def generate_3d_face(face_path):
    img = cv2.imread(face_path)
    if img is None:
        return None

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = vision.MpImage(
        image_format=vision.ImageFormat.SRGB,
        data=rgb
    )

    result = face_landmarker.detect(mp_image)
    if not result.face_landmarks:
        return None

    h, w, _ = img.shape
    xs, ys, zs = [], [], []

    for lm in result.face_landmarks[0]:
        xs.append(lm.x * w)
        ys.append(lm.y * h)
        zs.append(lm.z * w)

    fig = go.Figure(
        data=[go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode='markers',
            marker=dict(size=2, color=zs, colorscale='Viridis')
        )]
    )

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(
            xaxis_visible=False,
            yaxis_visible=False,
            zaxis_visible=False,
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        )
    )
    return fig

# ================= CAMERA =================
cap = cv2.VideoCapture(0)

# ================= MAIN LOOP =================
if run:
    status_box.success("Detection Running")

    while run:
        ret, frame = cap.read()
        if not ret:
            status_box.error("Camera not available")
            break

        frame_buffer.append(frame.copy())
        now = time.time()

        results = yolo(frame, conf=0.3, classes=[0], verbose=False)

        for r in results:
            for box in r.boxes:
                if now - last_capture_time < COOLDOWN:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if (x2 - x1) < MIN_BODY_W or (y2 - y1) < MIN_BODY_H:
                    continue

                best_body, best_score = None, 0
                for bf in list(frame_buffer)[-8:]:
                    body = bf[y1:y2, x1:x2]
                    if body.size == 0:
                        continue
                    s = sharpness(body)
                    if s > best_score:
                        best_score = s
                        best_body = body

                if best_body is None:
                    continue

                blob = cv2.dnn.blobFromImage(
                    cv2.resize(best_body, (300, 300)), 1.0,
                    (300, 300), (104, 177, 123)
                )

                face_net.setInput(blob)
                detections = face_net.forward()
                bh, bw = best_body.shape[:2]

                for i in range(detections.shape[2]):
                    if detections[0, 0, i, 2] < 0.6:
                        continue

                    fx1, fy1, fx2, fy2 = (
                        detections[0, 0, i, 3:7] * np.array([bw, bh, bw, bh])
                    ).astype(int)

                    face = best_body[fy1:fy2, fx1:fx2]
                    if face.size == 0 or sharpness(face) < FACE_BLUR_THRESHOLD:
                        continue

                    desc = face_descriptor(face)
                    if desc is None:
                        continue

                    ts = time.strftime("%Y%m%d_%H%M%S")
                    idx = -1
                    for j, kf in enumerate(known_faces):
                        if is_same_person(desc, kf["descriptor"]):
                            idx = j
                            break

                    if idx == -1:
                        face_path = os.path.join(BASE_DIR, f"captured/face/face_{ts}.jpg")
                        body_path = os.path.join(BASE_DIR, f"captured/body/person_{ts}.jpg")
                        cv2.imwrite(face_path, face)
                        cv2.imwrite(body_path, best_body)
                        known_faces.append({"descriptor": desc, "path": face_path})
                        speak_person_detected()

                    last_capture_time = now
                    break

        frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        lf = latest_image(os.path.join(BASE_DIR, "captured/face"))
        lb = latest_image(os.path.join(BASE_DIR, "captured/body"))

        if lf:
            face_placeholder.image(lf, caption="Last Face", width=220)
            fig = generate_3d_face(lf)
            if fig:
                face_3d_placeholder.plotly_chart(fig, use_container_width=True)

        if lb:
            body_placeholder.image(lb, caption="Last Body", width=220)

        time.sleep(0.03)

else:
    status_box.info("Detection Stopped")

cap.release()
