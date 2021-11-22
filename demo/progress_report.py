import numpy as np
import cv2
import pickle

cam_left = pickle.load(open("data/calibration/intrinsics_left.p", "rb"))
cam_right = pickle.load(open("data/calibration/intrinsics_right.p", "rb"))
rect_params = pickle.load(open("data/calibration/rectification_parameters.p", "rb"))
sgbm_params = pickle.load(open("data/calibration/sgbm_parameters.p", "rb"))
img_height, img_width = (480, 640)

R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
    cam_left['K'], cam_left['dist'], cam_right['K'],
    cam_right['dist'], (img_width, img_height), rect_params['R'],
    rect_params['T'])
map_left_1, map_left_2 = cv2.initUndistortRectifyMap(cam_left['K'], cam_left['dist'], R1, P1, (img_width, img_height), cv2.CV_16SC2)
map_right_1, map_right_2 = cv2.initUndistortRectifyMap(cam_right['K'], cam_right['dist'], R2, P2, (img_width, img_height), cv2.CV_16SC2)
stereo = cv2.StereoSGBM_create(sgbm_params["minDisparity"], sgbm_params["numDisparities"], sgbm_params["blockSize"],
                               sgbm_params["P1"], sgbm_params["P2"], sgbm_params["disp12MaxDiff"], sgbm_params["preFilterCap"],
                               sgbm_params["uniquenessRatio"], sgbm_params["speckleWindowSize"], sgbm_params["speckleRange"])

video_left = cv2.VideoCapture("data/original/depth_test_left.avi")
video_right = cv2.VideoCapture("data/original/depth_test_right.avi")
_, img_left= video_left.read()
_, img_right = video_right.read()
original = np.hstack((img_left, img_right))
for i in range(1, 24):
    cv2.line(original, (0, 20 * i), (1279, 20*i), 0)
cv2.imshow("original", original)
cv2.imwrite("demo/out/original.png", original)
rectified_left = cv2.remap(img_left, map_left_1, map_left_2, cv2.INTER_LINEAR)
rectified_right = cv2.remap(img_right, map_right_1, map_right_2, cv2.INTER_LINEAR)
rectified = np.hstack((rectified_left, rectified_right))
for i in range(1, 24):
    cv2.line(rectified, (0, 20 * i), (1279, 20*i), 0)
cv2.imshow("rectified", rectified)
cv2.imwrite("demo/out/rectified.png", rectified)
disparity = stereo.compute(rectified_left, rectified_right).astype(np.float32)
max, min = disparity.max(), disparity.min()
cv2.imshow("disparity", (disparity - min)/(max-min))
cv2.imwrite("demo/out/disparity.png", disparity)
cv2.waitKey(0)
