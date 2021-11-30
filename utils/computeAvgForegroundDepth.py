import numpy as np
import cv2

def computeAvgForegroundDepth(bgr_img, image_3d):
    img_height = bgr_img.shape[0]
    img_width = bgr_img.shape[1]
    x = int(img_width / 2) -15
    y = int(img_height / 2) -15
    w = 30
    h = 30

    cv2.rectangle(bgr_img, (x, y), (x + w, y + h), (255, 0, 255), 2)
    cv2.imshow('left', bgr_img)
    cv2.waitKey(30)

    depth_sum = 0
    x_sum = 0
    y_sum = 0
    num_pixels = 0

    for i in range(w):
        for j in range(h):
            if image_3d[x + i][y + j][2] != np.inf:
                num_pixels += 1
                x_sum += image_3d[x + i][y + j][0]
                y_sum += image_3d[x + i][y + j][1]
                depth_sum += image_3d[x + i][y + j][2]

    avg_depth = depth_sum / num_pixels
    avg_x = x_sum/num_pixels
    avg_y = y_sum / num_pixels
    # print("Num reasonable pixels: ", num_pixels)
    # print("x,y,z: ", (avg_x,avg_y,avg_depth))

    # depth_list = np.append(depth_list, avg_depth)
    # scipy.io.savemat('depth_list.mat', {'depth_list': depth_list})
    print("Average depth at square is: ", avg_depth)
    return bgr_img


