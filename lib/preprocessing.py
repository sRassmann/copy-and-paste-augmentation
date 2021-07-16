"""
functionality used in data preprocessing
"""
import os
import numpy as np
import cv2
import pandas as pd
import scipy
from glob import glob
import skimage
from skimage.filters import *
from tqdm import tqdm

def segment_bugs_from_crops(img, gaus_sigma = 1/40, average_blur_size=1/20):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # legs
    legs = gray.copy()
    legs_background = scipy.ndimage.gaussian_filter(
        legs, sigma=round(img.shape[1] * gaus_sigma)
    )
    legs = cv2.subtract(legs_background, legs)
    legs = (legs > threshold_otsu(legs)).astype(np.uint8)

    # body
    body = cv2.blur(gray, (round(img.shape[1] * average_blur_size),) * 2)
    body = body < threshold_minimum(body)
    body = scipy.ndimage.binary_fill_holes(body).astype(np.uint8)

    # whole bug
    close_size = max(7, img.shape[1] // 80)
    mask = cv2.bitwise_or(body, legs)
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    )
    mask = mask.astype(np.uint8)

    cont, hir = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    areas = np.array([cv2.contourArea(cnt) for cnt in cont])
    idx = areas.argmax()

    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.drawContours(
        image=mask,
        contours=cont,
        contourIdx=idx,
        color=(255),
        thickness=-1,
        hierarchy=hir,
        maxLevel=1,
    )
    return mask
    