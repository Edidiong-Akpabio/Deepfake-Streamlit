import cv2
import numpy as np
from config import IMG_WIDTH, IMG_HEIGHT

def resize_and_pad_image(img):
    old_size = img.shape[:2]
    ratio = min(IMG_WIDTH / old_size[1], IMG_HEIGHT / old_size[0])
    new_size = (int(old_size[1] * ratio), int(old_size[0] * ratio))
    img_resized = cv2.resize(img, new_size)

    delta_w = IMG_WIDTH - new_size[0]
    delta_h = IMG_HEIGHT - new_size[1]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    padded_img = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return padded_img

def prepare_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = resize_and_pad_image(img)
    img = img / 255.0
    return np.expand_dims(img, axis=0)
