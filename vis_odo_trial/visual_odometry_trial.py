import numpy as np
import cv2
import sys
from logitech_webcam_calibration import *

def getKMatrixForLogitechWebcams():
    f = 531
    c_x = 320
    c_y = 240
    K = np.array([[f, 0, c_x],
                  [0, f, c_y],
                  [0, 0, 1]])
    return K,K

def main():
    # Read images from a video file in the current folder.
    video_capture_left = cv2.VideoCapture('../data/original/binocular_test_left.avi')  # Open left video capture object
    video_capture_right = cv2.VideoCapture('../data/original/binocular_test_right.avi')  # Open right video capture object
    got_image_left, bgr_img_left = video_capture_left.read()  # Make sure we can read video from the left camera
    got_image_right, bgr_img_right = video_capture_right.read()  # Make sure we can read video from the right camera

    if not got_image_left or not got_image_right:
        print("Cannot read video source")
        sys.exit()

    frame_count = 0
    while True:
        got_image_left, bgr_img_left = video_capture_left.read()
        got_image_right, bgr_img_right = video_capture_right.read()

        if not got_image_left or not got_image_right:
            print("Breaking out :/")
            break  # End of video; exit the while loop
        img_left = cv2.cvtColor(bgr_img_left, cv2.COLOR_BGR2GRAY)
        img_right = cv2.cvtColor(bgr_img_right, cv2.COLOR_BGR2GRAY)
        img_height = img_left.shape[0]
        img_width = img_left.shape[1]
        print(img_width, img_height)
        # cv2.imshow('left_frames', bgr_img_left)
        # cv2.imshow('right_frames', bgr_img_right)
        # cv2.waitKey(30)

        # Load the left and right images in gray scale
        # imgLeft = cv2.imread('logga.png', 0)
        # imgRight = cv2.imread('logga1.png', 0)

        # Initialize the stereo block matching object
        stereo = cv2.StereoBM_create(numDisparities=16, blockSize=5)

        # Compute the disparity image
        disparity = stereo.compute(img_left, img_right)

        # Normalize the image for representation
        min = disparity.min()
        max = disparity.max()
        disparity = np.uint8(6400 * (disparity - min) / (max - min))

        # retval, K_left, distCoeffs_left, K_right, distCoeffs_right, R, T, E, F = getCalibrationParametersForLogitechWebcams()

        # Q = cv2.stereoRectify(cameraMatrix1=K_left,cameraMatrix2=K_right,
        #                       distCoeffs1 = distCoeffs_left,distCoeffs2 = distCoeffs_right,
        #                       R=R,T=T, imageSize= (img_width,img_height))

        # depth = cv2.reprojectImageTo3D(disparity, Q)
        # # Display the disparity map
        cv2.imshow('disparity map', np.hstack((img_left,img_right, disparity)))
        cv2.waitKey(30)
        # cv2.destroyAllWindows()



if __name__ == "__main__":
    main()