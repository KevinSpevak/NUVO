import cv2
from utils.rot2RPY import *
def drawPoseOnImage(T, img):
    roll, pitch, yaw = rot2RPY(T)
    t_x = float(T[0,3])
    t_y = float(T[1,3])
    t_z = float(T[2,3])
    # print('roll: ', roll, 'pitch: ', pitch,'yaw: ', yaw, 'x,y,z: ', t_x, t_y, t_z)
    image_height = img.shape[0]
    image_width = img.shape[1]
    pose_text = 'Yaw = (' + str(np.round(yaw[0], 3)) + ')' + \
                ' tx = ' + str(np.round(t_x, 1)) + ' ty = ' + str(np.round(t_y, 1)) + ' tz = ' + str(np.round(t_z, 1))
    print('pose text: ', pose_text)
    cv2.putText(img, text=pose_text, org=(0, image_height - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.45, color=(255, 255, 255))