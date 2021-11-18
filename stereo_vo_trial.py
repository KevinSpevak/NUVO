import cv2
import numpy as np
import sys
from utils.camera import *

video0 = cv2.VideoCapture("data/intermediates/hall0_flat.avi")
video1 = cv2.VideoCapture("data/intermediates/hall1_flat.avi")
cam0 = TUM_VI_CAM_0_PIN
cam1 = TUM_VI_CAM_1_PIN

success, im0 = video0.read()
_, im1 = video1.read()
if not success:
    sys.exit()
while success:
    im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)

    # Slight hack: assume both images were taken by the same camera in order to find the relative poses of
    # camera 0 with respect to camera 1. Then perform a perspective transformation on image 1 as if it were taken
    # from a camera orientation identical to camera 0. This serves as our "rectification" step
    R, t = find_relative_pose(cam0, im1, im0)
    map1, map2 = cv2.initUndistortRectifyMap(cam1.K(), cam1.dist, R, cam0.K(),
                                             (im0.shape[1], im0.shape[0]), cv2.CV_16SC2)
    rectified = cv2.remap(im1, map1, map2, cv2.INTER_LINEAR)
    for x in range(20):
        cv2.line(im0, (0, 25*x+5), (511, 25*x+5), (0, 0, 255))
        cv2.line(im1, (0, 25 * x + 5), (511, 25 * x + 5), (0, 0, 255))
        cv2.line(rectified, (0, 25 * x + 5), (511, 25 * x + 5), (0, 0, 255))
    cv2.imshow("camera 0", im0)
    cv2.imshow("camera 1", im1)
    cv2.imshow("rectified", rectified)
    matcher = cv2.StereoBM.create(160, 21)
    disparity = matcher.compute(im0, rectified)
    disp = cv2.normalize(disparity, None, alpha=0, beta=255,
                         norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imshow("disparity", disp)
    cv2.waitKey(0)
    break # TODO
    success, im0 = video0.read()
    _, im1 = video1.read()
