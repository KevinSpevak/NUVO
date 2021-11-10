import cv2
import os
import sys
import numpy as np
import glob


def main():
    # Define constants
    # Read images from the left camera
    video_capture_left = cv2.VideoCapture(1)  # Open video capture object

    # Read images from the right camera
    video_capture_right = cv2.VideoCapture(2)  # Open video capture object

    got_image_left, bgr_img_left = video_capture_left.read()  # Make sure we can read video from the left and right
    got_image_right, bgr_img_right = video_capture_right.read()

    if not got_image_left or got_image_right:
        print("Cannot read one of the video sources")
        sys.exit()

    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')

    videoWriter_left = cv2.VideoWriter("binocular_test_left.avi", fourcc=fourcc, fps=24.0,
                                  frameSize=(bgr_img_left.shape[1], bgr_img_left.shape[0]))
    videoWriter_right = cv2.VideoWriter("binocular_test_right.avi", fourcc=fourcc, fps=24.0,
                                       frameSize=(bgr_img_right.shape[1], bgr_img_right.shape[0]))

    while True:
        got_image_left, bgr_img_left = video_capture_left.read()
        got_image_right, bgr_img_right = video_capture_right.read()

        if not got_image_left or got_image_right:
            print("Breaking out :/")
            break  # End of video; exit the while loop

        videoWriter_left.write(bgr_img_left)
        videoWriter_right.write(bgr_img_right)
    videoWriter_left.release()
    videoWriter_right.release()


if __name__ == "__main__":
    main()