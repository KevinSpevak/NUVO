import cv2
import numpy as np
from utils.camera import StereoCameraPair
from utils.visual_odometry import Odometer

def main():
    video_left = cv2.VideoCapture("data/original/hall_left.avi")
    video_right = cv2.VideoCapture("data/original/hall_right.avi")
    stereo = StereoCameraPair.from_pfiles("data/calibration/intrinsics_left.p",
                                          "data/calibration/intrinsics_right.p",
                                          "data/calibration/rectification_parameters.p",
                                          "data/calibration/sgbm_parameters.p", (640, 480))
    odometer = Odometer(stereo)
    frame_num = 1

    while True:
        got_left, img_left = video_left.read()
        got_right, img_right = video_right.read()
        if not got_left and not got_right:
            break
        odometer.update(img_left, img_right)
        print("Frame num", frame_num)
        frame_num += 1
        print("current pose\n", odometer.current_pose())
        cv2.imshow("left", img_left)
        if cv2.waitKey(1) == 27:
            break
if __name__ == "__main__":
    main()
