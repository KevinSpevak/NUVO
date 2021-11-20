import cv2
import os
import sys
import numpy as np
import glob


def main():
    # Define constants
    # Read images from the left camera
    video_capture_left = cv2.VideoCapture(6)  # Open video capture object

    # Read images from the right camera
    video_capture_right = cv2.VideoCapture(2)  # Open video capture object

    got_image_left, bgr_img_left = video_capture_left.read()  # Make sure we can read video from the left and right
    got_image_right, bgr_img_right = video_capture_right.read()

    if not got_image_left or not got_image_right:
        print("Cannot read one of the video sources")
        sys.exit()

    recording = False
    while True:
        got_image_left, bgr_img_left = video_capture_left.read()
        got_image_right, bgr_img_right = video_capture_right.read()

        if not got_image_left or not got_image_right:
            print("Breaking out :/")
            break  # End of video; exit the while loop

        cv2.imshow("left", bgr_img_left)
        cv2.imshow("right", bgr_img_right)
        key = cv2.waitKey(40)
        if key == 13: # enter
            if recording:
                videoWriter_left.release()
                videoWriter_right.release()
                break;
            else:
                fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
                videoWriter_left = cv2.VideoWriter("data/calibration/stereo_left1.avi", fourcc=fourcc, fps=24.0,
                                                   frameSize=(bgr_img_left.shape[1], bgr_img_left.shape[0]))
                videoWriter_right = cv2.VideoWriter("data/calibration/stereo_right1.avi", fourcc=fourcc, fps=24.0,
                                                    frameSize=(bgr_img_right.shape[1], bgr_img_right.shape[0]))
                recording = True
        elif not recording and key == 27: # esc
            break
        if recording:
            videoWriter_left.write(bgr_img_left)
            videoWriter_right.write(bgr_img_right)

if __name__ == "__main__":
    main()
