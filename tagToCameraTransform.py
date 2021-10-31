import cv2
import numpy as np

def getCameraIntrinsicParameterMatrix():
# ToDo - Calibrate chosen camera and read in from calibration file
    f = 675
    c_x = 320
    c_y = 240
    K = np.array([[f, 0, c_x],
                  [0, f, c_y],
                  [0, 0, 1]], float)
    return K


def tagToCameraTransform(bgr_img):
    # Takes in video frames and returns the transformation to the camera

    translation = np.empty((4, 1))

    # Create a dictionary object
    arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)

    # Get the camera intrinsic parameters
    K = getCameraIntrinsicParameterMatrix()

    # Detect and draw the ArUco tags
    corners, ids, _ = cv2.aruco.detectMarkers(image=bgr_img, dictionary=arucoDict)

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(image=bgr_img, corners=corners, ids=ids,
                                                         borderColor=(0, 0, 255))

    # Detect the pose of the markers
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners=corners, markerLength=2.0,
                                                          cameraMatrix=K, distCoeffs=0)
    # Get the pose of the first detected marker with respect to the camera.
    if rvecs is not None:
        rvec_m_c = rvecs[0]                 # This is a 1x3 rotation vector
        tm_c = tvecs[0]                     # This is a 1x3 translation vector

        # Draw the pose
        cv2.aruco.drawAxis(image=bgr_img, cameraMatrix=K, distCoeffs=0, rvec=rvec_m_c, tvec=tm_c,
                           length=0.5)
# ToDo - Need new ids
#         if ids == 0:
#             # translation = np.array([[2.5],
#             #                         [-2.0],
#             #                         [-1.0],
#             #                        [1]])
#         elif ids == 1:
#             # translation = np.array([[-2.5],
#             #                         [-2.0],
#             #                         [-5.0],
#             #                        [1]])
#         else:
#            print('Unknown tag detected')

        # 3D-2D perspective projection used for debugging only
        # Need to use the camera extrinsic parameters matrix, ie, c_T_m and K to find the location of the switch
        # in image space
        # R, _ = cv2.Rodrigues(rvecs[0])
        # T = np.append(R, np.transpose(tvecs[0]), axis=1)
        # switch_coords = np.matmul(K, np.matmul(T, translation))
        # x = int(switch_coords[0]/switch_coords[2])
        # y = int(switch_coords[1]/switch_coords[2])
        # w = 10
        # h = 10
        #
        # cv2.rectangle(bgr_img, (x, y), (x + w, y + h), (255, 0, 255), 2)

    return rvecs, tvecs

