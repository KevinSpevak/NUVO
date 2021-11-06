import rosbag
import cv2
import numpy as np
import array

# jank decoding since utility function cv_bridge is not easily installable
def decode_ROS_img(img):
    int_length_bytes = int(img.step / img.width)
    max_val = 2**(8*int_length_bytes) * 1.0
    decoded = []
    for y in range(img.height):
        # hardcoding unsigned short type
        # might need to fix to handle different encodings/platform-specific int types
        row = array.array('H')
        row.frombytes(img.data[y*img.step:(y+1)*img.step])
        decoded.append(row)
    return np.array(decoded)/max_val*256

# assuming camera times are synced
def extract_from_bag(bagfile, outname):
    bag = rosbag.Bag("data/raw/" + bagfile)
    # hardcoding topic names for now; may need to take as arguments
    cam0_name = '/cam0/image_raw'
    cam1_name = '/cam1/image_raw'
    cam0_frames = bag.read_messages(topics=[cam0_name])
    cam1_frames = bag.read_messages(topics=[cam1_name])
    fps = bag.get_type_and_topic_info().topics[cam0_name].frequency
    out0 = None
    out1 = None

    while True:
        _, im0, _ = next(cam0_frames, (None, None, None))
        if im0 is None:
            break
        _, im1, _ = next(cam1_frames)
        if out0 is None:
           out0 = cv2.VideoWriter("data/extracted/" + outname + "0.avi", fourcc=cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                 fps=fps, frameSize=(im0.width, im0.height), isColor=False)
           out1 = cv2.VideoWriter("data/extracted/" + outname + "1.avi", fourcc=cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                 fps=fps, frameSize=(im0.width, im0.height), isColor=False)
        out0.write(decode_ROS_img(im0).astype('uint8'))
        out1.write(decode_ROS_img(im1).astype('uint8'))
    out0.release()
    out1.release()

extract_from_bag("dataset-corridor4_512_16.bag", "hall")