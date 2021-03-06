# Utility for finding a good set of options for OpenCV's StereoSGBM
import cv2
import pickle
import sys
import numpy as np

def find_sgbm_settings(calib_file_left, calib_file_right, rect_file, save_file, img_left, img_right):
    cam_left = pickle.load(open(calib_file_left, "rb"))
    cam_right = pickle.load(open(calib_file_right, "rb"))
    rect_params = pickle.load(open(rect_file, "rb"))
    img_height, img_width = img_left.shape

    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
        cam_left['K'], cam_left['dist'], cam_right['K'],
        cam_right['dist'], (img_width, img_height), rect_params['R'],
        rect_params['T'])
    map_left_1, map_left_2 = cv2.initUndistortRectifyMap(cam_left['K'], cam_left['dist'], R1, P1, (img_width, img_height), cv2.CV_16SC2)
    map_right_1, map_right_2 = cv2.initUndistortRectifyMap(cam_right['K'], cam_right['dist'], R2, P2, (img_width, img_height), cv2.CV_16SC2)
    rectified_left = cv2.remap(img_left, map_left_1, map_left_2, cv2.INTER_LINEAR)
    rectified_right = cv2.remap(img_right, map_right_1, map_right_2, cv2.INTER_LINEAR)

    # min disparity, num disparities/16, (block size - 1)/2, p1/(block size)^2,
    # p2/(3 * (block size)^2), max disparity, prefilter cap / 10, uniqueness ratio,
    # speckle window size / 10, speckle range
    sgbm_vars = [0, 16, 1, 0, 0, 0, 0, 10, 0, 0]
    stereo = cv2.StereoSGBM_create(0, 16, 1, 0, 0, 0, 0, 10, 0, 0)

    def display():
        disparity = stereo.compute(rectified_left, rectified_right).astype(np.float32)
        max, min = disparity.max(), disparity.min()
        cv2.imshow("disparity", (disparity - min)/(max-min))
    display()

    def update_min_disp(val):
        stereo.setMinDisparity(val)
        sgbm_vars[0] = val
        display()
    #cv2.createTrackbar("Min Disparity", "disparity", 0, 15, update_min_disp)

    def update_num_disp(val):
        stereo.setNumDisparities(16 * val)
        sgbm_vars[1] = 16 * val
        display()
    cv2.createTrackbar("Num Disparities", "disparity", 1, 20, update_num_disp)

    def update_block_size(val):
        stereo.setBlockSize(val * 2 + 1)
        sgbm_vars[2] = val * 2 + 1
        display()
    cv2.createTrackbar("Block Size", "disparity", 0, 10, update_block_size)

    def update_p1(val):
        stereo.setP1(val * sgbm_vars[2]**2)
        sgbm_vars[3] = val * sgbm_vars[2]**2
        display()
    cv2.createTrackbar("P1", "disparity", 0, 15, update_p1)

    def update_p2(val):
        stereo.setP2(3 * val * sgbm_vars[2]**2)
        sgbm_vars[4] = 3 * val * sgbm_vars[2]**2
        display()
    cv2.createTrackbar("P2", "disparity", 1, 25, update_p2)
    

    def update_max_disp(val):
        stereo.setDisp12MaxDiff(val)
        sgbm_vars[5] = val
        display()
    cv2.createTrackbar("Max Disparity", "disparity", 0, 50, update_max_disp)

    def update_prefilter_cap(val):
        stereo.setPreFilterCap(val * 10)
        sgbm_vars[6] = val * 10
        display()
    cv2.createTrackbar("Prefilter Cap", "disparity", 0, 10, update_prefilter_cap)

    def update_uniq_ratio(val):
        stereo.setUniquenessRatio(val)
        sgbm_vars[7] = val
        display()
    cv2.createTrackbar("Uniqueness Ratio", "disparity", 1, 20, update_uniq_ratio)
    
    def update_speckle_size(val):
        stereo.setSpeckleWindowSize(val * 10)
        sgbm_vars[8] = val * 10
        display()
    cv2.createTrackbar("Speckle Size", "disparity", 0, 25, update_speckle_size)

    def update_speckle_range(val):
        stereo.setSpeckleRange(val)
        sgbm_vars[9] = val
        display()
    cv2.createTrackbar("Speckle Range", "disparity", 0, 4, update_speckle_range)

    while True:
        if cv2.waitKey(0) == 27:
            break

    params = {"minDisparity": sgbm_vars[0], "numDisparities": sgbm_vars[1], "blockSize": sgbm_vars[2],
              "P1": sgbm_vars[3], "P2": sgbm_vars[4], "disp12MaxDiff": sgbm_vars[5],
              "preFilterCap": sgbm_vars[6], "uniquenessRatio": sgbm_vars[7],
              "speckleWindowSize": sgbm_vars[8], "speckleRange": sgbm_vars[9]}
    print(params)
    pickle.dump(params, open(save_file, "wb"))

if __name__ == "__main__":
    video_capture_left = cv2.VideoCapture('data/original/depth_test_left.avi')  # Open left video capture object
    video_capture_right = cv2.VideoCapture('data/original/depth_test_right.avi')  # Open right video capture object
    got_image_left, bgr_img_left = video_capture_left.read()  # Make sure we can read video from the left camera
    got_image_right, bgr_img_right = video_capture_right.read()  # Make sure we can read video from the right camera

    if not got_image_left or not got_image_right:
        print("Cannot read video source")
        sys.exit()

    bgr_img_left = cv2.cvtColor(bgr_img_left, cv2.COLOR_BGR2GRAY)
    bgr_img_right = cv2.cvtColor(bgr_img_right, cv2.COLOR_BGR2GRAY)
    find_sgbm_settings("data/calibration/intrinsics_left.p",
                       "data/calibration/intrinsics_right.p",
                       "data/calibration/rectification_parameters.p",
                       "data/calibration/sgbm_parameters.p",
                       bgr_img_left, bgr_img_right)
