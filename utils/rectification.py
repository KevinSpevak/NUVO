import os
import cv2
import numpy as np
import pickle

# Look for chessboard pattern in 1 out of this many video frames
CALIBRATION_VIDEO_FRAMES_PER_ITERATION = 10

# Generate rectification parameters from a stereo video feed of a chessboard
# requires calibration values (Intrinsic Matrix, Distortion Parameters) for each camera
def generate_rectification_parameters(video_file_left, video_file_right, calib_file_left, calib_file_right,
                                      out_file, chessboard_size, square_size):
    for file in [video_file_left, video_file_right, calib_file_left, calib_file_right]:
        assert(os.path.exists(file))
    video_left = cv2.VideoCapture(video_file_left)
    video_right = cv2.VideoCapture(video_file_right)
    calib_left = pickle.load(open(calib_file_left, "rb"))
    calib_right = pickle.load(open(calib_file_right, "rb"))
    cols, rows = chessboard_size
    obj_points = np.zeros((cols * rows, 3), np.float32)
    obj_points[:, :2] = square_size * np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    img_points_left, img_points_right = [], []
    img_size = None

    # criteria for refining corner locations
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    frame_count, processed_count, found_count = 0, 0, 0

    print("Finding chessboard corners in video frames...")
    while True:
        got_left, img_left = video_left.read()
        got_right, img_right = video_right.read()
        if not got_left or not got_right:
            break
        # Speed up by not considering every frame
        if frame_count % CALIBRATION_VIDEO_FRAMES_PER_ITERATION == 0:
            img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
            img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
            found_left, corners_left = cv2.findChessboardCorners(img_left, chessboard_size, None)
            found_right, corners_right = cv2.findChessboardCorners(img_right, chessboard_size, None)
            if (found_left and found_right):
                corners_left = cv2.cornerSubPix(img_left, corners_left, (11, 11), (-1, -1),criteria)
                corners_right = cv2.cornerSubPix(img_right, corners_right, (11, 11), (-1, -1), criteria)
                img_points_left.append(corners_left)
                img_points_right.append(corners_right)
                if not img_size:
                    img_size = tuple(reversed(img_left.shape))
                found_count += 1
            processed_count += 1
        frame_count += 1
    print("Frames: {} | Processed: {} | Found Chessboard: {}".format(frame_count, processed_count, found_count))
    print("Starting Rectification...")
    success, _, _, _, _, R, T, E, F = cv2.stereoCalibrate([obj_points] * found_count, img_points_left, img_points_right,
                                                          calib_left['K'], calib_left['dist'], calib_right['K'], calib_right['dist'],
                                                          img_size, flags=cv2.CALIB_FIX_INTRINSIC)
    print("R: ", R)
    print("T: ", T)
    print("E: ", E)
    print("F: ", F)
    pickle.dump({"R": R, "T": T, "E": E, "F": F}, open(out_file, "wb"))

if __name__ == "__main__":
    generate_rectification_parameters("data/calibration/stereo_left.avi", "data/calibration/stereo_right.avi",
                                      "data/calibration/intrinsics_left.p", "data/calibration/intrinsics_right.p",
                                      "data/calibration/rectification_parameters.p", (8, 6), 0.025)
