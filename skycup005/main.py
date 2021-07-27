import os
import cv2
import glob
import numpy as np
import tensorflow as tf
import core.utils as utils
from core.tracker import Tracker

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def iou(bbox1, bbox2):
    bbox1 = [float(x) for x in bbox1]
    bbox2 = [float(x) for x in bbox2]
    (x0_1, y0_1, x1_1, y1_1) = bbox1
    (x0_2, y0_2, x1_2, y1_2) = bbox2
    overlap_x0 = max(x0_1, x0_2)
    overlap_y0 = max(y0_1, y0_2)
    overlap_x1 = min(x1_1, x1_2)
    overlap_y1 = min(y1_1, y1_2)
    if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
        return 0
    size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
    size_union = size_1 + size_2 - size_intersection
    return size_intersection / size_union

def convertBack(xmin, ymin, xmax, ymax):
    return [xmin, ymin, xmax, ymax]

return_elements = ['input/input_data:0', 'pred_sbbox/concat_2:0', 'pred_mbbox/concat_2:0', 'pred_lbbox/concat_2:0']
pb_file = 'yolov3.pb'
graph = tf.Graph()
return_tensors = utils.read_pb_return_tensors(graph, pb_file, return_elements)
feature_params = dict( maxCorners = 20,qualityLevel = 0.3,minDistance = 7, blockSize = 21 )
lk_params = dict(winSize  = (15,15),maxLevel = 2,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

num_classes = 1
input_size = 640

with tf.compat.v1.Session(graph=graph) as sess:
    folders = os.listdir('/data/input_data')
    for folder in folders:
        lines = []
        images = glob.glob(os.path.join('/data/input_data', folder, '*.bmp'))
        images = sorted(images)
        tracks_active = []
        tracker = Tracker(160, 100, 1)
        frame_num = 0
        move = np.array([0,0])
        for image in images:
            image_data = cv2.imread(image)
            if frame_num == 0:
                old_gray = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
                p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
            else:
                frame_gray = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
                p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
                good_new = p1[st == 1]
                good_old = p0[st == 1]
                oldlist = []
                newlist = []
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    oldlist.append([c, d])
                    newlist.append([a,b])
                oldlist = np.array(oldlist)
                newlist = np.array(newlist)
                speed = np.mean(newlist - oldlist, 0)
                move = move + speed
                old_gray = frame_gray.copy()
                p0 = good_new.reshape(-1, 1, 2)
                # print(move)
                if abs(move[0]) > 40 or abs(move[1]) > 40:
                    # old_gray = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
                    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
            frame_num += 1
            image_size = image_data.shape[:2]
            input_data = utils.image_preporcess(image_data.copy(), [input_size, input_size])
            input_data = input_data[np.newaxis, ...]
            pred_sbbox, pred_mbbox, pred_lbbox = sess.run([return_tensors[1], return_tensors[2], return_tensors[3]], feed_dict={ return_tensors[0]: input_data})
            pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)), np.reshape(pred_mbbox, (-1, 5 + num_classes)), np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)
            bboxes = utils.postprocess_boxes(pred_bbox, image_size, input_size, 0.3)
            bboxes = utils.nms(bboxes, 0.45, method='nms')
            bboxes = [convertBack(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])) for bbox in bboxes if float(bbox[4]) > 0.2]
            centers = []
            for bbox in bboxes:
                centers.append([[(bbox[0] + bbox[2]) / 2 - move[0]], [(bbox[1] + bbox[3]) / 2 - move[1]]])
            if (len(centers) > 0):
                tracker.Update(centers)
                for i in range(len(tracker.tracks)):
                    if tracker.tracks[i].flag:
                        cv2.putText(image_data, str(tracker.tracks[i].track_id), (int(tracker.tracks[i].prediction[0][0] + move[0]),
                                                                              int(tracker.tracks[i].prediction[1][0] + move[1])), 0, 5e-3 * 130, (0, 255, 0), 2)

            cv2.namedWindow("show", cv2.WINDOW_NORMAL)
            cv2.imshow('show', image_data)
            if not os.path.exists(image.replace('input', 'output')[:-7]):
                os.mkdir(image.replace('input', 'output')[:-7])
            cv2.imwrite(image.replace('input', 'output'), image_data)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
