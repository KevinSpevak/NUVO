import numpy as np
import cv2
import glob
import sys

# This program reads images of a chessboard, finds corners, and calibrates the camera.
# The chessboard is assumed to be 7 rows and 9 columns of squares.
# Create points in target coordinates; i.e., (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0).
# These are the inner corners of the squares. Note that it doesn't find the outer corners, so the actual
# grid is 6 x 8 corners. Units don't matter because we are not interested in the absolute camera poses.
grid_size = 0.025 # m
target_pts = np.zeros((6 * 8, 3), np.float32)
target_pts[:, :2] = grid_size*np.mgrid[0:8, 0:6].T.reshape(-1, 2)
print(target_pts)
# Arrays to store object points and image points from all the images.
objpoints = []  # Collect all 3d points in target coordinates
imgpointsLeft = []  # Collect all 2d points from the left camera in image plane
imgpointsRight = []  # Collect all 2d points from the right camera in image plane

# Read images from a video file in the current folder.
video_capture_left = cv2.VideoCapture('../data/original/calibrate_left.avi')  # Open left video capture object
video_capture_right = cv2.VideoCapture('../data/original/calibrate_right.avi')  # Open right video capture object
got_image_left, bgr_img_left = video_capture_left.read()  # Make sure we can read video from the left camera
got_image_right, bgr_img_right = video_capture_right.read()  # Make sure we can read video from the right camera

if not got_image_left or not got_image_right:
    print("Cannot read video source")
    sys.exit()

frame_count = 0
while True:
    got_image_left, bgr_img_left = video_capture_left.read()
    got_image_right, bgr_img_right = video_capture_right.read()
    if frame_count % 10 == 0:
        print("frame_count: ", frame_count)

        if not got_image_left or not got_image_right:
            print("Breaking out :/")
            break  # End of video; exit the while loop
        # img = cv2.imread(fname)
        gray_image_left = cv2.cvtColor(bgr_img_left, cv2.COLOR_BGR2GRAY)
        gray_image_right = cv2.cvtColor(bgr_img_right, cv2.COLOR_BGR2GRAY)
        h, w = gray_image_left.shape
        # Find the chess board corners
        ret_val_left, corners_left = cv2.findChessboardCorners(gray_image_left, (8, 6), None)
        ret_val_right, corners_right = cv2.findChessboardCorners(gray_image_right, (8, 6), None)
        # If found, add object and image points.
        if ret_val_left == True and ret_val_right == True:
            # Optionally refine corner locations.
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2_left = cv2.cornerSubPix(gray_image_left, corners_left, (11, 11), (-1, -1),criteria)
            corners2_right = cv2.cornerSubPix(gray_image_right, corners_right, (11, 11), (-1, -1), criteria)
            # Collect the object and image points.
            objpoints.append(target_pts)
            imgpointsLeft.append(corners2_left)
            imgpointsRight.append(corners2_right)
            # Draw and display the corners
            cv2.drawChessboardCorners(bgr_img_left, (6, 8), corners2_left, ret_val_left)
            cv2.drawChessboardCorners(bgr_img_right, (6, 8), corners2_right, ret_val_right)
            cv2.imshow('img_left', bgr_img_left)
            cv2.imshow('img_right', bgr_img_right)
            # print("Image points left: ", imgpointsLeft)
            cv2.waitKey(10)

    frame_count = frame_count + 1
cv2.destroyAllWindows()
# Do the calibration.
print("starting calibration")
retval, K_left, distCoeffs_left, K_right, distCoeffs_right, R, T, E, F = cv2.stereoCalibrate(
    objectPoints=objpoints, imagePoints1=imgpointsLeft, imagePoints2=imgpointsRight, imageSize=(w, h),
    cameraMatrix1=None, distCoeffs1=None, cameraMatrix2=None, distCoeffs2=None)
print("ret_val: ", retval)

print("Camera matrix 1:")
print(repr(K_left))
print("Camera matrix 2:")
print(repr(K_right))
print("Distortion coeffs 1:")
print(repr(distCoeffs_left))
print("Distortion coeffs 2:")
print(repr(distCoeffs_right))
# Calculate re-projection error - should be close to zero.
mean_error_left = 0
mean_error_right = 0
rvecs,_ = cv2.Rodrigues(R)
print("rvecs", rvecs)
print("T", T)
for i in range(len(objpoints)):
    imgpoints_left, _ = cv2.projectPoints(objpoints[i], rvecs, T, K_left, distCoeffs_left)
    imgpoints_right, _ = cv2.projectPoints(objpoints[i], rvecs, T, K_right, distCoeffs_right)
    error_left = cv2.norm(imgpointsLeft[i], imgpoints_left, cv2.NORM_L2)/len(imgpoints_left)
    error_right = cv2.norm(imgpointsRight[i], imgpoints_right, cv2.NORM_L2) / len(imgpoints_right)
    mean_error_left += error_left
    mean_error_right += error_right
print( "total left error: {}".format(mean_error_left/len(objpoints)))
print( "total right error: {}".format(mean_error_right/len(objpoints)))
# Optionally undistort and display the images.
frame_count = 0
video_capture_left = cv2.VideoCapture('../data/original/calibrate_left.avi')  # Open left video capture object
video_capture_right = cv2.VideoCapture('../data/original/calibrate_right.avi')  # Open right video capture object
while True:
    got_image_left, bgr_img_left = video_capture_left.read()
    got_image_right, bgr_img_right = video_capture_right.read()

    if frame_count % 10 == 0:
        print("frame_count: ", frame_count)
        cv2.imshow("distorted_left", bgr_img_left)
        cv2.imshow("distorted_right", bgr_img_right)
        undistorted_img_left = cv2.undistort(src=bgr_img_left, cameraMatrix=K_left, distCoeffs=distCoeffs_left)
        undistorted_img_right = cv2.undistort(src=bgr_img_right, cameraMatrix=K_right, distCoeffs=distCoeffs_right)
        cv2.imshow("undistorted_left", undistorted_img_left)
        cv2.imshow("undistorted_right", undistorted_img_right)
        # cv2.imwrite("undistorted_" + fname, undistorted_img)
        if cv2.waitKey(0) == 27:  # ESC is ascii code 27
            break
    frame_count += 1