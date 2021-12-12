import cv2
import numpy as np
from utils.camera import StereoCameraPair
from utils.visual_odometry import Odometer
from utils.drawPoseOnImage import *


def main():
    video_left = cv2.VideoCapture("data/original/hall_left.avi")
    video_right = cv2.VideoCapture("data/original/hall_right.avi")
    stereo = StereoCameraPair.from_pfiles("data/calibration/intrinsics_left.p",
                                          "data/calibration/intrinsics_right.p",
                                          "data/calibration/rectification_parameters.p",
                                          "data/calibration/sgbm_parameters.p", (640, 480))
    odometer = Odometer(stereo)
    frame_num = 1

    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')

    videoWriter = cv2.VideoWriter("final_demo_2.avi", fourcc=fourcc, fps=24.0,
                                  frameSize=(640, 480))
    while True and frame_num < 171:
        got_left, img_left = video_left.read()
        got_right, img_right = video_right.read()
        if not got_left and not got_right:
            break
        odometer.update(img_left, img_right)
        print("Frame num", frame_num)
        frame_num += 1
        print("current pose\n", odometer.current_pose())
        # Print disparity map with pose

        max, min = odometer.current_disparity.max(), odometer.current_disparity.min()
        # cv2.imshow("disparity", (self.current_disparity - min)/(max-min))
        normed_disp = (odometer.current_disparity - min) / (max - min)
        three_dimensional_disparity = cv2.cvtColor(normed_disp, cv2.COLOR_GRAY2RGB)
        drawPoseOnImage(odometer.current_pose(), img_left)
        cv2.imshow('image with pose', img_left)
        cv2.waitKey(30)
        videoWriter.write(img_left)

        cv2.imshow("left", img_left)
        if cv2.waitKey(1) == 27:
            break
    videoWriter.release()
if __name__ == "__main__":
    main()
