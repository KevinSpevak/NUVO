import numpy as np
import cv2
import sys
import pickle

def main():
    # Read images from a video file in the current folder.
    video_capture_left = cv2.VideoCapture('../data/original/depth_test_left.avi')  # Open left video capture object
    video_capture_right = cv2.VideoCapture('../data/original/depth_test_right.avi')  # Open right video capture object
    got_image_left, bgr_img_left = video_capture_left.read()  # Make sure we can read video from the left camera
    got_image_right, bgr_img_right = video_capture_right.read()  # Make sure we can read video from the right camera

    if not got_image_left or not got_image_right:
        print("Cannot read video source")
        sys.exit()

    img_height = bgr_img_left.shape[0]
    img_width = bgr_img_left.shape[1]
    # print(img_width, img_height)
    frame_count = 0

    # Left and right camera intrinsic parameters and stereo rectification parameters
    cam_left = pickle.load(open("../data/calibration/intrinsics_left.p", "rb"))
    cam_right = pickle.load(open("../data/calibration/intrinsics_right.p", "rb"))
    rect_params = pickle.load(open("../data/calibration/rectification_parameters.p", "rb"))
    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(cam_left['K'], cam_left['dist'], cam_right['K'], cam_right['dist'],
                                                                      (img_width, img_height), rect_params['R'], rect_params['T'])
    while True:
        got_image_left, bgr_img_left = video_capture_left.read()
        got_image_right, bgr_img_right = video_capture_right.read()

        if not got_image_left or not got_image_right:
            print("Breaking out :/")
            break  # End of video; exit the while loop
        img_left = cv2.cvtColor(bgr_img_left, cv2.COLOR_BGR2GRAY)
        img_right = cv2.cvtColor(bgr_img_right, cv2.COLOR_BGR2GRAY)

        # Initialize the stereo block matching object
        stereo = cv2.StereoBM_create(numDisparities=16, blockSize=5)

        # Compute the disparity image
        disparity = stereo.compute(img_left, img_right)

        # Normalize the image for representation
        min = disparity.min()
        max = disparity.max()
        # disparity = np.uint8(6400 * (disparity - min) / (max - min))
        # Display the disparity map
        # cv2.imshow('disparity map', np.hstack((img_left, img_right, disparity)))
        # cv2.waitKey(30)
        if cv2.waitKey(0) == 27:  # ESC is ascii code 27
            break

        image_3d = cv2.reprojectImageTo3D(disparity, Q)
        cv2.imshow('3D image', image_3d)
        cv2.waitKey(30)
        print("Disparity shape: ", disparity.shape)
        print("Q shape: ", Q.shape)
        print("What is a 3D Image: ", image_3d[320,240])
    # cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
