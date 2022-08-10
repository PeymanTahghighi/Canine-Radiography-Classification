import cv2
import numpy as np


if __name__ == "__main__":
    img = cv2.imread("281.jpeg", cv2.IMREAD_GRAYSCALE);
    thresh = cv2.threshold(img, thresh=150, maxval=250, type=cv2.THRESH_BINARY)[1];
    cv2.imshow('t', thresh);
    cv2.waitKey();