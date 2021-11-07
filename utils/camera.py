import cv2
import numpy as np
import sys

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

# undistort_video("hall0.avi", "hall0_flat.avi", TUM_VI_CAM_0, TUM_VI_CAM_0_PIN)
# undistort_video("hall1.avi", "hall1_flat.avi", TUM_VI_CAM_1, TUM_VI_CAM_1_PIN)
