import streamlit as st
import cv2
import mediapipe as mp
from math import hypot
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import numpy as np
import time

# Initialize Mediapipe and Audio control
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
vol_min, vol_max = volume.GetVolumeRange()[:2]

# Streamlit app setup
st.title("Hand Gesture Volume Control")
st.write("Control your system volume by adjusting the distance between your thumb and index finger.")

# Initialize session state for run control
if "run" not in st.session_state:
    st.session_state["run"] = False

# Capture video from webcam
cap = cv2.VideoCapture(0)

# Sidebar for Volume and Display options
with st.sidebar:
    st.header("Settings")
    current_volume = st.slider(
        "Current Volume",
        value=float(volume.GetMasterVolumeLevel()),  # Set as float
        min_value=vol_min,
        max_value=vol_max,
        step=0.1  # Define step as float
    )
    display_hands = st.checkbox("Display Hand Landmarks")

# Button to start/stop the video stream
def start_video():
    st.session_state["run"] = True

def stop_video():
    st.session_state["run"] = False
    cap.release()

st.button("Start Volume Control", on_click=start_video)
st.button("Stop Volume Control", on_click=stop_video)

# Placeholder for video frames
frame_placeholder = st.empty()

# Main loop to process video frames
while st.session_state["run"]:
    success, img = cap.read()
    if not success:
        st.write("Failed to capture video.")
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    lm_list = []
    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            for id, lm in enumerate(hand_landmark.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])
            if display_hands:
                mp_draw.draw_landmarks(img, hand_landmark, mp_hands.HAND_CONNECTIONS)

    if lm_list:
        x1, y1 = lm_list[4][1], lm_list[4][2]  # Thumb tip
        x2, y2 = lm_list[8][1], lm_list[8][2]  # Index tip

        cv2.circle(img, (x1, y1), 4, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, (x2, y2), 4, (255, 0, 0), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

        # Calculate length between thumb and index finger
        length = hypot(x2 - x1, y2 - y1)

        # Map the length to volume range
        vol = np.interp(length, [15, 220], [vol_min, vol_max])
        volume.SetMasterVolumeLevel(vol, None)

    # Update volume slider based on actual volume with unique key
    st.sidebar.slider(
        "Current Volume (Updated)",
        value=float(volume.GetMasterVolumeLevel()),  # Set as float
        min_value=vol_min,
        max_value=vol_max,
        step=0.1,  # Define step as float
        key="updated_volume_slider"
    )

    # Display the frame in Streamlit
    frame_placeholder.image(img, channels="BGR")

    # Control the frame rate
    time.sleep(0.05)

st.write("Click 'Start Volume Control' to begin.")
