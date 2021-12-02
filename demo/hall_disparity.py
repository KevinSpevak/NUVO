import pickle
import cv2
import numpy as np
from utils.computeAvgForegroundDepth import *

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
    orb = cv2.ORB_create()
    matcher = cv2.BFMatcher.create(cv2.NORM_HAMMING)
    video_left = cv2.VideoCapture("data/original/wall_left.avi")
    video_right = cv2.VideoCapture("data/original/wall_right.avi")
    prev_img_left, prev_kps, prev_img_3d = None, None, None
    cumulative = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    while True:
        got_left, img_left = video_left.read()
        got_right, img_right = video_right.read()
        if not got_left and not got_right:
            break
        img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
        rectified_left = cv2.remap(img_left, map_left_1, map_left_2, cv2.INTER_LINEAR)
        rectified_right = cv2.remap(img_right, map_right_1, map_right_2, cv2.INTER_LINEAR)

        # Magic number 16 used here - collective comprehension for the reason is low
        disparity = stereo.compute(rectified_left, rectified_right).astype(np.float32)/16
        max, min = disparity.max(), disparity.min()
        cv2.imshow("disparity", (disparity - min)/(max-min))
        cv2.imshow("right", img_right)
        cv2.imshow("left", img_left)
        img_3d = cv2.reprojectImageTo3D(disparity, Q)

        # Get features
        kps, desc = orb.detectAndCompute(img_left, None)
        if (prev_kps):
            matches = matcher.knnMatch(prev_desc, desc, k=2)
            # TODO ambiguous match threshold
            matches = [m[0] for m in matches if m[0].distance < 0.8 * m[1].distance]
            match_img = cv2.drawMatches(prev_img_left, prev_kps, img_left, kps, matches, None)
            cv2.imshow("matches", match_img)
            print([kps[m.trainIdx].pt for m in matches])
            pts_3d = [img_3d[int(y)][int(x)] for x, y in [kps[m.trainIdx].pt for m in matches]]
            prev_pts_3d = [prev_img_3d[int(y)][int(x)] for x, y in [prev_kps[m.queryIdx].pt for m in matches]]
            # TODO ignore features outside of valid ROI?
            success, T, inliers = cv2.estimateAffine3D(np.array(prev_pts_3d), np.array(pts_3d))

            if (success):
                print(T)
                print("inliers: ", sum(inliers))
            else:
                print("failure")

        prev_img_left = img_left # TODO needed?
        prev_kps, prev_desc = kps, desc
        prev_img_3d = img_3d

        img_left = computeAvgForegroundDepth(rectified_left, image_3d)
        cv2.imshow('Calculating depth at box', img_left)

        if cv2.waitKey(40) == 27:
            break

if __name__ == "__main__":
    main()
