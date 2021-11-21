import numpy as np
import cv2
import sys
import pickle
ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

def main():
    # Read images from a video file in the current folder.
    video_capture_left = cv2.VideoCapture('../data/original/depth_1m_left.avi')  # Open left video capture object
    video_capture_right = cv2.VideoCapture('../data/original/depth_1m_right.avi')  # Open right video capture object
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

    # disparity range is tuned for 'aloe' image pair
    window_size = 3
    min_disp = 5
    num_disp = 21 - min_disp
    stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                                   numDisparities=num_disp,
                                   blockSize=3,
                                   P1=8 * 3 * window_size ** 2,
                                   P2=32 * 3 * window_size ** 2,
                                   disp12MaxDiff=1,
                                   uniquenessRatio=10,
                                   speckleWindowSize=100,
                                   speckleRange=32
                                   )
    print('computing disparity...')


    while True:
        got_image_left, bgr_img_left = video_capture_left.read()
        got_image_right, bgr_img_right = video_capture_right.read()
        image_left = cv2.cvtColor(bgr_img_left, cv2.COLOR_BGR2GRAY)
        image_right = cv2.cvtColor(bgr_img_right, cv2.COLOR_BGR2GRAY)
        img_height = image_left.shape[0]
        img_width = image_left.shape[1]
        print("Input image dim: ",img_width, img_height)
        if not got_image_left or not got_image_right:
            print("Breaking out :/")
            break  # End of video; exit the while loop

        disparity = stereo.compute(image_left, image_right).astype(np.float32)# / 16.0
        print('Disparity shape: ', disparity.shape)
        image_3d = cv2.reprojectImageTo3D(disparity, Q)


        # image_3d= cv2.cvtColor(image_3d, cv2.COLOR_BGR2GRAY)
        cv2.imshow('3D image', image_3d)
        cv2.waitKey(0)

        # cv2.imshow('disparity', (disparity - min_disp) / num_disp)

        print('3D image shape: ',image_3d.shape)
        # print('3rd dim : ', image_3d[2])
        #print("Calculating depth at: ", (320,240))
        cv2.drawMarker(bgr_img_left, position=(320, 240),
                       color=(0,0,0), markerType=cv2.MARKER_CROSS)
        cv2.imshow('left', bgr_img_left)
        cv2.waitKey(0)
        # print("Value at (320,240,1) is: ", image_3d[320][240][0])
        # print("Value at (320,240,2) is: ", image_3d[320][240][1])
        print("Depth at (320,240,3) is: ", image_3d[320][240][2])




if __name__ == "__main__":
    main()
