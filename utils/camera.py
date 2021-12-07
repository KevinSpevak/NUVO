import cv2
import numpy as np
import sys
import pickle

class StereoCameraPair:
    @classmethod
    def from_pfiles(cls, left_cam_file, right_cam_file, rect_file, sgbm_file, img_size):
        cam_left = pickle.load(open(left_cam_file, "rb"))
        cam_right = pickle.load(open(right_cam_file, "rb"))
        rect_params = pickle.load(open(rect_file, "rb"))
        sgbm_params = pickle.load(open(sgbm_file, "rb"))
        return cls(cam_left['K'], cam_left['dist'], cam_right['K'],
                   cam_right['dist'], rect_params, sgbm_params, img_size)

    def __init__(self, K_left, dist_left, K_right, dist_right, rect_params, sgbm_params, img_size):
        R1, R2, P1, P2, self.Q, self.valid_region_left, self.valid_region_right = cv2.stereoRectify(
            K_left, dist_left, K_right, dist_right, img_size, rect_params['R'], rect_params['T'])
        self.map_left_1, self.map_left_2 = cv2.initUndistortRectifyMap(
            K_left, dist_left, R1, P1, img_size, cv2.CV_16SC2)
        self.map_right_1, self.map_right_2 = cv2.initUndistortRectifyMap(
            K_right, dist_right, R2, P2, img_size, cv2.CV_16SC2)
        self.stereoSGBM = cv2.StereoSGBM_create(
            sgbm_params["minDisparity"], sgbm_params["numDisparities"], sgbm_params["blockSize"],
            sgbm_params["P1"], sgbm_params["P2"], sgbm_params["disp12MaxDiff"], sgbm_params["preFilterCap"],
            # TODO: mode=1 still needed?
            0 and sgbm_params["uniquenessRatio"], sgbm_params["speckleWindowSize"], sgbm_params["speckleRange"], mode=1)
        print("sgbm",
              sgbm_params["minDisparity"], sgbm_params["numDisparities"], sgbm_params["blockSize"],
              sgbm_params["P1"], sgbm_params["P2"], sgbm_params["disp12MaxDiff"], sgbm_params["preFilterCap"],
              0 and sgbm_params["uniquenessRatio"], sgbm_params["speckleWindowSize"], sgbm_params["speckleRange"])

    def undistort_rectify_left(self, img):
        return cv2.remap(img, self.map_left_1, self.map_left_2, cv2.INTER_LINEAR)

    def undistort_rectify_right(self, img):
        return cv2.remap(img, self.map_right_1, self.map_right_2, cv2.INTER_LINEAR)

    def crop_to_valid_region_left(self, img):
        return img[self.valid_region_left[1]: self.valid_region_left[3],
                   self.valid_region_left[0]: self.valid_region_left[2]]

    def crop_to_valid_region_right(self, img):
        return img[self.valid_region_right[1]: self.valid_region_right[3],
                   self.valid_region_right[0]: self.valid_region_right[2]]

    def compute_3d(self, img_left, img_right):
        img_left = self.undistort_rectify_left(cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY))
        img_right = self.undistort_rectify_right(cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY))
        disparity = self.stereoSGBM.compute(img_left, img_right).astype(np.float32)/16
        disp_img = (disparity - disparity.min())/(disparity.max()-disparity.min())
        img_3d = cv2.reprojectImageTo3D(disparity, self.Q)
        return (self.crop_to_valid_region_left(img_3d),
                self.crop_to_valid_region_left(disparity),
                self.crop_to_valid_region_left(img_left))

### DEPRICATED ###

# Model of a camera with intrinsic parameters
# Note xi is only for omnidirectional camera model (probably won't use for webcams)
class Camera:
    def __init__(self, fx, fy, cx, cy, dist, xi=None):
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        self.dist, self.xi = np.array(dist), np.array(xi)

    def K(self):
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])

    # undistort img assuming it was taken with this camera
    # target_cam is a camera with the desired intrinsics (returns image as if it were taken by target_cam)
    def undistort(self, img, target_cam):
        if self.xi is None:
            return img # TODO pinhole camera model undistortion
        else:
            return cv2.omnidir.undistortImage(img, self.K(), self.dist, self.xi,
                                              cv2.omnidir.RECTIFY_PERSPECTIVE, Knew=target_cam.K())

# Intrinsics for cameras used in the TUM VI dataset (not sure about x/y order)
# Omnidirectional Model
TUM_VI_CAM_0 = Camera(533.340727445877, 533.2556495307942, 254.64689387916482, 256.4835490935692,
                     [-0.05972430882700243, 0.17468739202093328, 0.000737218969875311, 0.000574074894976456],
                      1.7921879013)
TUM_VI_CAM_1 = Camera(520.2546241208013, 520.1799003708908, 252.24978846121377, 254.15045097300418,
                      [-0.07693518083211431, 0.12590335598238764, 0.0016421936053305271, 0.0006230553630283544],
                      1.73241756065)
# Pinhole model
TUM_VI_CAM_0_PIN = Camera(190.97847715128717, 190.9733070521226, 254.9317060593547, 256.8974428996504,
                          [0.0034823894022493434, 0.0007150348452162257, -0.0020532361418706202, 0.00020293673591811182])
TUM_VI_CAM_1_PIN = Camera(190.44236969414825, 190.4344384721956, 252.59949716835982, 254.91723064636983,
                          [0.0034003170790442797, 0.001766278153469831, -0.00266312569781606, 0.0003299517423931039])

class ImageProcessingError(Exception):
    def __init__(self, message):
        self.message = message

def undistort_video(original, outname, source_cam, target_cam):
    video = cv2.VideoCapture("../data/extracted/" + original)
    success, frame = video.read()
    if not success:
        sys.exit()
    out = cv2.VideoWriter("../data/intermediates/" + outname, fourcc=cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                          fps=video.get(cv2.CAP_PROP_FPS), frameSize=(frame.shape[1], frame.shape[0]))
    while success:
        out.write(source_cam.undistort(frame, target_cam))
        success, frame = video.read()
    out.release()

# Find the change in rotation and translation of a camera between two consecutive images
# Returns the rotation matrix and translation vector from the camera's pose when im0 was taken
# to the pose when im1 was taken
def find_relative_pose(cam, im0, im1):
    # hard-coding threshold for ambiguous matches for now
    ambiguous_match_thresh = 0.8
    # Detect Features
    orb = cv2.ORB_create()
    im0_kps, im0_desc = orb.detectAndCompute(im0, None)
    im1_kps, im1_desc = orb.detectAndCompute(im1, None)
    # Match Features in im1 (query) to im0 (train)
    matches = cv2.BFMatcher.create(cv2.NORM_HAMMING).knnMatch(im1_desc, im0_desc, k=2)
    matches = [m[0] for m in matches if m[0].distance < ambiguous_match_thresh * m[1].distance]
    if len(matches) < 5:
        raise ImageProcessingError("Not enough feature matches")
    im0_points = np.float32([im0_kps[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    im1_points = np.float32([im1_kps[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    # Find Essential Matrix and Rotation and Translation from camera pose 0 to pose 1
    E, _ = cv2.findEssentialMat(im0_points, im1_points, cam.K())
    success, R, t, _ = cv2.recoverPose(E, im0_points, im1_points, cam.K())
    if not success:
        raise ImageProcessingError("Could not recover pose")
    return R, t

# returns rectified images
# Not currently working, so currently just transforming the second image to match the perspective of the first
def rectify(cam0, cam1, im0, im1):
    # Card-coding these parameters for now (see cv2.findFundamentalMat)
    # Maximum distance of an inlier point from an epipolar line, documentation suggests 1-3
    ransacReprojThreshold = 3
    # Desired confidence that estimated matrix is correct. Can try increasing if we get
    # too many wrong values, or decreasing if this runs too slow
    confidence = 0.99

    # Detect Features
    orb = cv2.ORB_create()
    im0_kps, im0_desc = orb.detectAndCompute(im0, None)
    im1_kps, im1_desc = orb.detectAndCompute(im1, None)
    # Match Features (hard-coding 0.8 as ambiguous match threshold)
    matches = cv2.BFMatcher.create(cv2.NORM_HAMMING).knnMatch(im1_desc, im0_desc, k=2)
    matches = [m[0] for m in matches if m[0].distance < 0.8 * m[1].distance]
    if len(matches) < 8:
        raise ImageProcessingError("Not enough feature matches")
    im0_points = np.float32([im0_kps[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    im1_points = np.float32([im1_kps[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    # Find the fundamental matrix and compute rectification transforms
    F, mask = cv2.findFundamentalMat(im0_points, im1_points, cv2.FM_RANSAC, ransacReprojThreshold, confidence)
    # TODO: mask?
    success, H0, H1 = cv2.stereoRectifyUncalibrated(im0_points, im1_points, F, (im0.shape[1], im0.shape[0]))
    if not success:
        raise ImageProcessingError("could not find rectification transforms")
    print(H0)
    print(H1)
    im0_points = np.array([im0_points[i] for i in range(len(mask)) if mask[i]])
    im1_points = np.array([im1_points[i] for i in range(len(mask)) if mask[i]])
    success, H0, H1 = cv2.stereoRectifyUncalibrated(im0_points, im1_points, F, (im0.shape[1], im0.shape[0]))
    if not success:
        raise ImageProcessingError("could not find rectification transforms")
    print(H0)
    print(H1)



# undistort_video("hall0.avi", "hall0_flat.avi", TUM_VI_CAM_0, TUM_VI_CAM_0_PIN)
# undistort_video("hall1.avi", "hall1_flat.avi", TUM_VI_CAM_1, TUM_VI_CAM_1_PIN)
