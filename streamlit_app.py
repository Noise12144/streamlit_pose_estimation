import cv2
import mediapipe as mp
import requests
import av
import matplotlib.pyplot as plt
import numpy as np
import torch
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

#Detection Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', device='cpu') 
model.classes = 0 #Solo le persone
model.conf = 0.65 #threshold

#Pose Estimation Model
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) 
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)


def process(image):
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = model(image) #Detection
    df = result.pandas().xyxy[0] #Output

    #Ritaglio delle persone:
    for i, row in df.iterrows():
        xmin = int(df.loc[i, 'xmin'])
        xmax = int(df.loc[i, 'xmax'])
        ymin = int(df.loc[i, 'ymin'])
        ymax = int(df.loc[i, 'ymax'])
        temp_frame = image[ymin:ymax, xmin:xmax]
        results_holistic = holistic.process(temp_frame)
        results_hands = hands.process(temp_frame)
        # Draw landmark annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        #cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
        mp_drawing.draw_landmarks(
            image[ymin:ymax, xmin:xmax],
            results_holistic.face_landmarks,
            mp_holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_contours_style())
        mp_drawing.draw_landmarks(
            image[ymin:ymax, xmin:xmax],
            results_holistic.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles
            .get_default_pose_landmarks_style())
        if results_hands.multi_hand_landmarks:
          for hand_landmarks in results_hands.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image[ymin:ymax, xmin:xmax],
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles
                .get_default_hand_landmarks_style())

    return cv2.flip(image, 1)

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)
class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        #Funzione:        
        img = process(img)

        #Return video
        return av.VideoFrame.from_ndarray(img, format="bgr24")
webrtc_ctx = webrtc_streamer(
    key="WYH",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=VideoProcessor,
    async_processing=True,
)
