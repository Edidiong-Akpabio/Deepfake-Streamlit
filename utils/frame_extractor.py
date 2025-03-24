import cv2
import numpy as np
from config import IMG_HEIGHT, IMG_WIDTH

def extract_frames(video_path, num_frames):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        return np.array([])

    for i in np.linspace(0, total_frames - 1, num_frames, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT)) / 255.0
            frames.append(frame)
    cap.release()
    return np.array(frames)
