import pickle
import cv2
import numpy as np

def main():
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

    video_left = cv2.VideoCapture("data/original/hall_left.avi")
    video_right = cv2.VideoCapture("data/original/hall_right.avi")
    while True:
        got_left, img_left = video_left.read()
        got_right, img_right = video_right.read()
        if not got_left and not got_right:
            break
        img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
        rectified_left = cv2.remap(img_left, map_left_1, map_left_2, cv2.INTER_LINEAR)
        rectified_right = cv2.remap(img_right, map_right_1, map_right_2, cv2.INTER_LINEAR)
        disparity = stereo.compute(rectified_left, rectified_right).astype(np.float32)
        max, min = disparity.max(), disparity.min()
        cv2.imshow("disparity", (disparity - min)/(max-min))

        if cv2.waitKey(40) == 27:
            break

if __name__ == "__main__":
    main()