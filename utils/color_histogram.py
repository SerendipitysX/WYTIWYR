import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import cv2 as cv2

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_vector(image, bins=128):
    mask = image[:, :, 3]
    image = image[:, :, :3]
    red = cv2.calcHist(
        [image], [0], mask, [bins], [0, 256]
    )
    green = cv2.calcHist(
        [image], [1], mask, [bins], [0, 256]
    )
    blue = cv2.calcHist(
        [image], [2], mask, [bins], [0, 256]
    )
    vector = np.concatenate([red, green, blue], axis=0)
    vector = vector.reshape(-1)
    return vector


def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def cosine_sum(a, b):
    span = len(a)/3
    ra = a[:span]
    rb = b[:span]
    ga = a[span:span*2]
    gb = b[span:span*2]
    ba = a[span*2:]
    bb = b[span*2:]
    r = cosine(ra, rb) + cosine(ga, gb) + cosine(ba, bb)
    return r/3


def extract_color_hist(img):
    img1 = img.convert("RGBA")
    img1 = np.array(img1)
    query_vector = get_vector(img1)
    return query_vector
