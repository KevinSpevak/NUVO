import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
#from utils.computeAvgForegroundDepth import *

def computeAvgForegroundDepth(bgr_img, image_3d, disparity):
    img_height = bgr_img.shape[0]
    img_width = bgr_img.shape[1]
    x = int(img_width / 2) -15
    y = int(img_height / 2) -15
    w = 30
    h = 30

    cv2.rectangle(bgr_img, (x, y), (x + w, y + h), (255, 0, 255), 2)
    cv2.imshow('left depth', bgr_img)
    max, min = disparity.max(), disparity.min()
    disp_img = (disparity - min)/(max-min)
    minidisp = disparity[y:y+h, x:x+h]
    mini3d = image_3d[y:y+h, x:x+h]
#    cv2.imshow("mini disp", minidisp)
#    cv2.imshow("mini 3d", mini3d)
    cv2.rectangle(disp_img, (x, y), (x + w, y + h), (255, 0, 255), 2)
    cv2.imshow("disp depth", disp_img)

    miniz = mini3d[:,:,2]
    print("disp: min", minidisp.min(), "max", minidisp.max())
    print("3d: min", miniz.min(), "max", miniz.max())
    fig, axes = plt.subplots(1, 2)
    if (mini3d.min() > -np.inf and mini3d.max() < np.inf):
        axes[0].hist(minidisp.flatten(), bins=[x for x in range(int(minidisp.min()), int(minidisp.max()) +1)])
        axes[1].hist(miniz, bins=np.array([x for x in range(int(miniz.min()), int(mini3d.max())*10 +1)])/10)
        plt.show()
        1
    else:
        import pdb
        pdb.set_trace()

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


def valid_disparity(query_pt, query_disparity, train_pt, train_disparity, max_change=10):
    q_x, q_y = query_pt
    t_x, t_y = train_pt
    q_d = query_disparity[int(q_y)][int(q_x)]
    t_d = train_disparity[int(t_y)][int(t_x)]
    # TODO: don't hardcode these values?
    return q_d > 0 and q_d < 63 and t_d > 0 and t_d < 63  and abs(t_d - q_d) <= max_change

#def valid_3d_pt_mask(disp_img):

def main():
    cam_left = pickle.load(open("../data/calibration/intrinsics_left.p", "rb"))
    cam_right = pickle.load(open("../data/calibration/intrinsics_right.p", "rb"))
    rect_params = pickle.load(open("../data/calibration/rectification_parameters.p", "rb"))
    sgbm_params = pickle.load(open("../data/calibration/sgbm_parameters.p", "rb"))
    img_height, img_width = (480, 640)

    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
        cam_left['K'], cam_left['dist'], cam_right['K'],
        cam_right['dist'], (img_width, img_height), rect_params['R'],
        rect_params['T'])
    map_left_1, map_left_2 = cv2.initUndistortRectifyMap(cam_left['K'], cam_left['dist'], R1, P1, (img_width, img_height), cv2.CV_16SC2)
    map_right_1, map_right_2 = cv2.initUndistortRectifyMap(cam_right['K'], cam_right['dist'], R2, P2, (img_width, img_height), cv2.CV_16SC2)
    stereo = cv2.StereoSGBM_create(sgbm_params["minDisparity"], sgbm_params["numDisparities"], sgbm_params["blockSize"],
                                   sgbm_params["P1"], sgbm_params["P2"], sgbm_params["disp12MaxDiff"], sgbm_params["preFilterCap"],
                                   sgbm_params["uniquenessRatio"], sgbm_params["speckleWindowSize"], sgbm_params["speckleRange"], mode=1)
    orb = cv2.ORB_create()
    matcher = cv2.BFMatcher.create(cv2.NORM_HAMMING)
    video_left = cv2.VideoCapture("../data/original/wall_left.avi")
    video_right = cv2.VideoCapture("../data/original/wall_right.avi")
    prev_img_left, prev_kps, prev_img_3d, prev_disparity = None, None, None, None
    cumulative = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
#    T_prev = np.eye(4)
    skipped_frames = 0
    frame_num = 0

    skip_num = 0
    for x in range(skip_num):
        video_left.read()
        video_right.read()
        frame_num += 1
    while True:
        got_left, img_left = video_left.read()
        got_right, img_right = video_right.read()
        frame_num += 1
        if not got_left and not got_right:
            break
        img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
        img_left = cv2.remap(img_left, map_left_1, map_left_2, cv2.INTER_LINEAR)
        img_right = cv2.remap(img_right, map_right_1, map_right_2, cv2.INTER_LINEAR)

        # Magic number 16 used here - collective comprehension for the reason is low

        disparity = stereo.compute(img_left, img_right).astype(np.float32)/16
        max, min = disparity.max(), disparity.min()
        disp_img = (disparity - min)/(max-min)
        cv2.imshow("disparity", disp_img)

        #cv2.imshow("right", img_right)
        #cv2.imshow("left", img_left)
        img_3d = cv2.reprojectImageTo3D(disparity, Q)
        img_3d = img_3d[validPixROI1[1]: validPixROI1[3], validPixROI1[0]:validPixROI1[2]]
        img_left = img_left[validPixROI1[1]: validPixROI1[3], validPixROI1[0]:validPixROI1[2]]
        disparity = disparity[validPixROI1[1]: validPixROI1[3], validPixROI1[0]:validPixROI1[2]]

        #cv2.imshow("cropped left", img_left)
        


        # Get features
        kps, desc = orb.detectAndCompute(img_left, None)
        if (prev_kps):
            matches = matcher.knnMatch(prev_desc, desc, k=2)
            # TODO ambiguous match threshold
            matches = [m[0] for m in matches if m[0].distance < 0.8 * m[1].distance]
            print("unambiguous matches", len(matches))
            max_disparity_change = 10 * (1 + skipped_frames)
            #match_img1 = cv2.drawMatches(prev_img_left, prev_kps, img_left, kps, matches, None)
            #cv2.imshow("unambiguous matches", match_img1)

            matches = [m for m in matches if valid_disparity(prev_kps[m.queryIdx].pt, prev_disparity, kps[m.trainIdx].pt, disparity)]
            print("valid matches", len(matches))

            max, min = disparity.max(), disparity.min()
            disp_img = cv2.cvtColor((disparity - min)/(max-min), cv2.COLOR_GRAY2BGR)
            for m in matches:
                x, y = kps[m.trainIdx].pt
                cv2.drawMarker(disp_img, (int(x), int(y)), [0, 255, 0])
            cv2.imshow("disparity", disp_img)

            #dd = abs(disparity - prev_disparity)
            #print("Disparity Diff | min:", dd.min(), "max:", dd.max(), "mean:", dd.mean(), "median:", np.median(dd), "std dev:", dd.std())
            
            #match_img = cv2.drawMatches(prev_img_left, prev_kps, img_left, kps, matches, None)
            #cv2.imshow("valid matches", match_img)

#            bgr_disp = cv2.cvtColor(disp_img, cv2.COLOR_GRAY2BGR)
#            im3d_copy = img_3d.copy()
#            for x, y in [kps[m.trainIdx].pt for m in matches]:
#                if np.linalg.norm(img_3d[int(y)][int(x)]) > 1000:
#                    print("coords", (x, y))
#                    print("disp value:", disparity[int(y)][int(x)])
#                    print("3d point", img_3d[int(y)][int(x)])
#                    cv2.drawMarker(bgr_disp, (int(x), int(y)), (0, 0, 255))
#                    cv2.drawMarker(img_left, (int(x), int(y)), (0, 0, 255))
#                    cv2.drawMarker(im3d_copy, (int(x), int(y)), (0, 0, 0))
#                    cv2.imshow("disp", bgr_disp)
#                    cv2.imshow("left", img_left)
#                    cv2.imshow("3d", im3d_copy)
#                    cv2.waitKey(0)

            pts_3d = [img_3d[int(y)][int(x)] for x, y in [kps[m.trainIdx].pt for m in matches]]
            prev_pts_3d = [prev_img_3d[int(y)][int(x)] for x, y in [prev_kps[m.queryIdx].pt for m in matches]]

            # TODO: consider higher point requirement
            if (len(matches) >= 3):
                T, scale = cv2.estimateAffine3D(np.array(prev_pts_3d), np.array(pts_3d), force_rotation=True)
                T = np.vstack([T, [0,0,0,1]])
#                print("prev_T_next: ", T)
            else:
                # skip frame
                print("### TOO FEW MATCHES TO GENERATE TRANSFORMATION ###")
                T = np.array([np.nan])

            

            if False and skipped_frames > 0:
#                for row in range(len(dd)):
#                    for col in range(len(dd[0])):
#                        d1 = disparity[row][col]
#                        d2 = disparity[row][col]
#                        if d1 <= 0 or d2 <= 0 or d1 == 63 or d2 == 63:
#                            dd[row][col] = np.inf
#                        dd = dd.flatten()
#                        dd = dd[dd != np.inf]
                for row in range(len(dd)):
                    for col in range(len(dd[0])):
                        d1 = disparity[row][col]
                        d2 = prev_disparity[row][col]
                        if d1 <= 0 or d2 <= 0 or d1 == 63 or d2 == 63:
                            dd[row][col] = np.inf
                dd2 = dd.flatten()
                dd = dd[dd != np.inf]
                plt.hist(dd, bins=[i for i in range(64)])
                plt.show()

            print("frame num:", frame_num)

#            pts_3d = np.array(pts_3d)
#            prev_pts_3d = np.array(prev_pts_3d)
#            fig, axes = plt.subplots(1, 2)
#            if (pts_3d.min() > -np.inf and pts_3d.max() < np.inf):
#                axes[0].hist(pts_3d.flatten(), bins=[x for x in range(int(pts_3d.min()), int(pts_3d.max()) +1)])
#                axes[1].hist(prev_pts_3d.flatten(), bins=[x for x in range(int(pts_3d.min()), int(pts_3d.max()) +1)])
#                plt.show()
#                cv2.waitKey(0)
#            else:
#                import pdb
#                pdb.set_trace()

            if frame_num % 100 == 0:
                cv2.waitKey(0)

            if np.isnan(T).any():
                skipped_frames += 1
            else:
                skipped_frames = 0

            if skipped_frames > 0:
                T_prev = np.eye(4)
                print("Panic at the library!")
                print("T:\n", T)
                cv2.waitKey(0)

            else:
                cumulative = cumulative @ T
                print("cumulative:\n", cumulative)
#                T_prev = T_prev @ T
#                print("T_prev:\n", T_prev)
#                print("T:\n", T)

        if skipped_frames == 0:
            prev_img_left = img_left # TODO needed?
            prev_kps, prev_desc = kps, desc
            prev_img_3d = img_3d
            prev_disparity = disparity

        computeAvgForegroundDepth(img_left, img_3d, disparity)
        #cv2.imshow('Calculating depth at box', img_left)

        if cv2.waitKey(0) == 27:
            break

if __name__ == "__main__":
    main()
