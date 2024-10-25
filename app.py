import streamlit as st
import cv2
import mediapipe as mp
from math import hypot
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import numpy as np
import time

# Initialize Mediapipe for hand tracking and audio control
mp_hands = mp.solutions.hands  # Mediapipe Hands module
hands = mp_hands.Hands()  # Instantiate the Hands object
mp_draw = mp.solutions.drawing_utils  # Drawing utilities for hand landmarks

# Initialize audio control using Pycaw
devices = AudioUtilities.GetSpeakers()  # Get audio devices
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)  # Activate audio interface
volume = cast(interface, POINTER(IAudioEndpointVolume))  # Cast to IAudioEndpointVolume
vol_min, vol_max = volume.GetVolumeRange()[:2]  # Get the minimum and maximum volume levels

# Streamlit app setup
st.title("Hand Gesture Volume Control")  # Set the title of the app
st.write("Control your system volume by adjusting the distance between your thumb and index finger.")  # Instructions

# Initialize session state for controlling video stream
if "run" not in st.session_state:
    st.session_state["run"] = False  # Control variable to track if video is running

# Capture video from the webcam
cap = cv2.VideoCapture(0)  # Open the default camera

# Define button functions to start/stop video stream
def start_video():
    st.session_state["run"] = True  # Set the run state to True to start the loop

def stop_video():
    st.session_state["run"] = False  # Set the run state to False to stop the loop
    cap.release()  # Release the video capture object

# Create buttons to start and stop volume control
st.button("Start Volume Control", on_click=start_video)
st.button("Stop Volume Control", on_click=stop_video)

# Placeholder for video frames to be displayed in Streamlit
frame_placeholder = st.empty()

# Main loop to process video frames
while st.session_state["run"]:
    success, img = cap.read()  # Read a frame from the video capture
    if not success:
        st.write("Failed to capture video.")  # Handle case where frame capture fails
        break
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert the image from BGR to RGB for Mediapipe processing
    results = hands.process(img_rgb)  # Process the image to find hand landmarks

    lm_list = []  # List to store landmark positions
    if results.multi_hand_landmarks:  # Check if any hand landmarks were found
        for hand_landmark in results.multi_hand_landmarks:
            for id, lm in enumerate(hand_landmark.landmark):
                h, w, _ = img.shape  # Get the height and width of the frame
                cx, cy = int(lm.x * w), int(lm.y * h)  # Calculate the (x, y) coordinates of the landmark
                lm_list.append([id, cx, cy])  # Append the landmark ID and coordinates to the list
            mp_draw.draw_landmarks(img, hand_landmark, mp_hands.HAND_CONNECTIONS)  # Draw hand connections on the image

    if lm_list:  # If hand landmarks are detected
        x1, y1 = lm_list[4][1], lm_list[4][2]  # Get coordinates of the thumb tip
        x2, y2 = lm_list[8][1], lm_list[8][2]  # Get coordinates of the index finger tip

        # Draw circles and line connecting thumb and index finger
        cv2.circle(img, (x1, y1), 4, (255, 0, 0), cv2.FILLED)  # Draw thumb circle
        cv2.circle(img, (x2, y2), 4, (255, 0, 0), cv2.FILLED)  # Draw index finger circle
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)  # Draw line between thumb and index finger

        # Calculate the length between thumb and index finger
        length = hypot(x2 - x1, y2 - y1)  # Compute the Euclidean distance

        # Map the length to the volume range and set the master volume level
        vol = np.interp(length, [15, 220], [vol_min, vol_max])  # Interpolate volume level based on distance
        volume.SetMasterVolumeLevel(vol, None)  # Set the system volume

    # Display the processed frame in Streamlit
    frame_placeholder.image(img, channels="BGR")  # Show the image in the Streamlit app

    # Control the frame rate of the loop
    time.sleep(0.05)  # Pause for a short duration to control processing speed

# Final message for user
st.write("Click 'Start Volume Control' to begin.")  # Prompt to start the application
