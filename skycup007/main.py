#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
import warnings
import os
import cv2
import glob
import random
import numpy as np
from ctypes import *


def sample(probs):
    s = sum(probs)
    probs = [a / s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs) - 1


def c_array(ctype, values):
    arr = (ctype * len(values))()
    arr[:] = values
    return arr


class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]


class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]


class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]


lib = CDLL("model_data/libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)


def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2))) - 1
    xmax = int(round(xmin + w)) + 1
    ymin = int(round(y - (h / 2))) - 1
    ymax = int(round(ymin + h)) + 1
    return [xmin, ymin, xmax, ymax]


def array_to_image(arr):
    # need to return old values to avoid python freeing memory
    arr = arr.transpose(2, 0, 1)
    c, h, w = arr.shape[0:3]
    arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
    data = arr.ctypes.data_as(POINTER(c_float))
    im = IMAGE(w, h, c, data)
    return im, arr


def detect(net, meta, image, thresh=.1, hier_thresh=.5, nms=.4):
    im, image = array_to_image(image)
    rgbgr_image(im)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh,
                             hier_thresh, None, 0, pnum)
    num = pnum[0]
    if nms: do_nms_obj(dets, num, meta.classes, nms)

    confidences = []
    boxes = []
    labels = []
    for j in range(num):
        a = dets[j].prob[0:meta.classes]
        if any(a):
            ai = np.array(a).nonzero()[0]
            for i in ai:
                b = dets[j].bbox
                confidences.append(dets[j].prob[i])
                boxes.append([b.x, b.y, b.w, b.h])
                labels.append(i)
    return labels, boxes, confidences

warnings.filterwarnings('ignore')


def track_iou(detections, tracks_active, tracks_loss, sigma_l, sigma_h, sigma_iou, t_life, t_loss, id, framenum):
    """
    Simple IOU based tracker.
    See "High-Speed Tracking-by-Detection Without Using Image Information by E. Bochinski, V. Eiselein, T. Sikora" for
    more information.
    Args:
         detections (list): list of detections per frame
         tracks_active (list): list of tracks_active
         sigma_l (float): low detection threshold.
         sigma_h (float): high detection threshold.
         sigma_iou (float): IOU threshold.
         t_life (float): minimum track length in frames.
    Returns:
        list: list of tracks_finished and tracks_active.
    """
    tracks_finished = []

    # apply low threshold to detections
    dets = [det for det in detections if det['score'] >= sigma_l]
    del_dets = []

    updated_tracks = []
    for track in tracks_active:
        match_flag = False
        if len(dets) > 0:
            # get det with highest iou
            best_match = max(dets, key=lambda x: iou(track['bboxes'], x['bbox']))
            if iou(track['bboxes'], best_match['bbox']) >= sigma_iou:
                track['bboxes'] = best_match['bbox']
                track['max_score'] = max(track['max_score'], best_match['score'])
                track['life_time'] += 1
                track['loss_time'] = 0
                updated_tracks.append(track)

                # remove from best matching detection from detections
                match_flag = True
                del_dets.append(best_match)
                del dets[dets.index(best_match)]

                if track['max_score'] >= sigma_h and track['life_time'] >= t_life:
                    if track['id'] == 0:
                        flag = True
                        for track_loss in tracks_loss:
                            dis_id1_id2 = framenum - track_loss['framenum']
                            if dis_id1_id2 > 150:
                                continue
                            end_x = (track_loss['bboxes'][0] + track_loss['bboxes'][2]) / 2
                            end_y = (track_loss['bboxes'][1] + track_loss['bboxes'][3]) / 2
                            start_x = (track_loss['original_bbox'][0] + track_loss['original_bbox'][2]) / 2
                            start_y = (track_loss['original_bbox'][1] + track_loss['original_bbox'][3]) / 2
                            id_1_vec = np.array([end_x-start_x, end_y-start_y])
                            in_1_tt_num = track_loss['framenum'] - track_loss['original_framenum']
                            estimate_loc = np.array([end_x, end_y]) + dis_id1_id2 * id_1_vec/in_1_tt_num
                            track_x = (track['bboxes'][0] + track['bboxes'][2]) / 2
                            track_y = (track['bboxes'][1] + track['bboxes'][3]) / 2
                            estimate_vec = np.array([track_x, track_y]) - estimate_loc
                            move_dis = np.sqrt(estimate_vec.dot(estimate_vec))  # 根据运动速度估计第一个向量终点与起点距离
                            if move_dis < 100:
                                track['id'] = track_loss['id']
                                flag = False
                                tracks_loss.remove(track_loss)
                                break
                        if flag:
                            id += 1
                            track['id'] = id
                    tracks_finished.append(track)

        if not match_flag and len(del_dets) > 0 and track['id'] != 0:
            best_match = max(del_dets, key=lambda x: iou(track['bboxes'], x['bbox']))
            if iou(track['bboxes'], best_match['bbox']) >= sigma_iou:
                track['bboxes'] = best_match['bbox']
                track['max_score'] = max(track['max_score'], best_match['score'])
                track['life_time'] += 1
                track['loss_time'] = 0
                for x_track in tracks_finished:
                    if x_track['bboxes'] == track['bboxes']:
                        x_track['x_flag'] = True
                        track['x_flag'] = True
                for x_track in updated_tracks:
                    if x_track['bboxes'] == track['bboxes']:
                        x_track['x_flag'] = True
                        track['x_flag'] = True
                updated_tracks.append(track)

                # remove from best matching detection from detections
                del del_dets[del_dets.index(best_match)]

                if track['max_score'] >= sigma_h and track['life_time'] >= t_life:
                    tracks_finished.append(track)

        # if track was not updated
        if len(updated_tracks) == 0 or track is not updated_tracks[-1]:
            track['loss_time'] += 1
            if track['loss_time'] > 15:
                track['life_time'] = 0
            # finish track when the conditions are met
            if track['loss_time'] >= t_loss:
                track['framenum'] = framenum
                tracks_active.remove(track)
                if track['id'] != 0:
                    tracks_loss.append(track)

    # create new tracks
    new_tracks = [{'bboxes': det['bbox'], 'original_bbox':det['bbox'], 'original_framenum': framenum, 'framenum': framenum,
                   'max_score': det['score'], 'life_time':1, 'loss_time': 0, 'id': 0, 'x_flag': False} for det in dets]
    tracks_active = tracks_active + new_tracks

    x_tracks_index = [tracks_finished.index(track) for track in tracks_finished if track['x_flag']]
    if len(x_tracks_index) == 2:
        x_track1 = tracks_finished[x_tracks_index[0]]
        x_track2 = tracks_finished[x_tracks_index[1]]
        index = max(distance(x_track1, x_track1), distance(x_track2, x_track2), distance(x_track1, x_track2),
                    distance(x_track2, x_track1))
        if index < 2:
            tracks_finished[x_tracks_index[0]]['x_flag'] = False
            tracks_finished[x_tracks_index[1]]['x_flag'] = False
        else:
            _id = tracks_finished[x_tracks_index[0]]['id']
            tracks_finished[x_tracks_index[0]]['id'] = tracks_finished[x_tracks_index[1]]['id']
            tracks_finished[x_tracks_index[1]]['id'] = _id
            tracks_finished[x_tracks_index[0]]['x_flag'] = False
            tracks_finished[x_tracks_index[1]]['x_flag'] = False

    return tracks_finished, tracks_active, tracks_loss, id

def distance(track1, track2):
    track1_end_x = (track1['bboxes'][0] + track1['bboxes'][1]) / 2
    track1_end_y = (track1['bboxes'][2] + track1['bboxes'][3]) / 2

    track2_start_x = (track2['original_bbox'][0] + track2['original_bbox'][1]) / 2
    track2_start_y = (track2['original_bbox'][2] + track2['original_bbox'][3]) / 2

    return abs(track1_end_x - track2_start_x) + abs(track1_end_y - track2_start_y)


def iou(bbox1, bbox2):
    """
    Calculates the intersection-over-union of two bounding boxes.
    Args:
        bbox1 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.
        bbox2 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.
    Returns:
        int: intersection-over-onion of bbox1, bbox2
    """

    bbox1 = [float(x) for x in bbox1]
    bbox2 = [float(x) for x in bbox2]

    (x0_1, y0_1, x1_1, y1_1) = bbox1
    (x0_2, y0_2, x1_2, y1_2) = bbox2

    # get the overlap rectangle
    overlap_x0 = max(x0_1, x0_2)
    overlap_y0 = max(y0_1, y0_2)
    overlap_x1 = min(x1_1, x1_2)
    overlap_y1 = min(y1_1, y1_2)

    # check if there is an overlap
    if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
        return 0

    # if yes, calculate the ratio of the overlap to each ROI size and the unified size
    size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
    size_union = size_1 + size_2 - size_intersection

    return size_intersection / size_union


def main():
    net = load_net(b"model_data/SIA.cfg", b"model_data/SIA_final.weights", 0)
    meta = load_meta(b"model_data/SIA.data")
    feature_params = dict(maxCorners=20, qualityLevel=0.3, minDistance=7, blockSize=21)
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5, 0.03))

    folders = os.listdir("/data/input_data/")
    for folder in folders:
        images = glob.glob(os.path.join("/data/input_data/", folder, "*.bmp"))
        images = sorted(images)

        lines = []
        tracks_active = []
        tracks_loss = []
        id = 0

        frame_num = 0
        move = np.array([0,0])
        move_img = np.zeros((250, 2), dtype=np.int)
        for image in images:
            frame = cv2.imread(image)

            if frame_num == 0:
                old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
            else:
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
                good_new = p1[st == 1]
                good_old = p0[st == 1]
                oldlist = []
                newlist = []
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    oldlist.append([c, d])
                    newlist.append([a, b])
                oldlist = np.array(oldlist)
                newlist = np.array(newlist)
                speed = np.mean(newlist - oldlist, 0)
                move = (move + speed).astype(np.int)
                old_gray = frame_gray.copy()
                p0 = good_new.reshape(-1, 1, 2)
                move_img[frame_num, :] = move
                # print(move)
                if abs(move[0]) > 40 or abs(move[1]) > 40:
                    # old_gray = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
                    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)



            labels, boxes, confidences = detect(net, meta, frame)

            boxes = [convertBack(float(box[0]-move[0]), float(box[1]-move[1]), float(box[2]), float(box[3])) for box in boxes]
            # for _box in boxes:
            #     cv2.rectangle(frame, (int(_box[0] + move[0]), int(_box[1] + move[1])),
            #                   (int(_box[2] + move[0]), int(_box[3] + move[1])), (0, 255, 0), 2)

            detections = [{'bbox': bbox, 'score': confidence} for bbox, confidence in zip(boxes, confidences)]

            tracks_finished, tracks_active, tracks_loss, id = track_iou(detections, tracks_active, tracks_loss, 0.1, 0.2, 0.1, 10, 20, id, int(image[-7:-4]))


            for track_finished in tracks_finished:
                bbox = track_finished['bboxes']
                cv2.rectangle(frame,(int(bbox[0] + move[0]), int(bbox[1] + move[1])), (int(bbox[2] + move[0]), int(bbox[3] + move[1])),(255,0,0), 2)
                cv2.putText(frame, str(track_finished['id']), (int(bbox[0] + move[0]), int(bbox[3] + move[1])), 0, 5e-3 * 130, (0,255,0),2)

            cv2.namedWindow("show", cv2.WINDOW_NORMAL)
            cv2.imshow('show', frame)
            if not os.path.exists(image.replace('input', 'output')[:-7]):
                os.mkdir(image.replace('input', 'output')[:-7])
            cv2.imwrite(image.replace('input', 'output'), frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        #     if len(tracks_finished) == 0:
        #         line = 'frame:' + image[-7:-4] + ' ' + str(len(tracks_finished))
        #     else:
        #         line = 'frame:' + image[-7:-4] + ' ' + str(len(tracks_finished))
        #         for track_finished in tracks_finished:
        #             bbox = track_finished['bboxes']
        #             cen_x = int((bbox[0] + bbox[2]) / 2) + move[0]
        #             cen_y = int(((bbox[1] + bbox[3]) / 2)) + move[1]
        #             line = line + ' ' + 'object:' + str(track_finished['id']) + ' ' + str(cen_x) + ' ' + str(cen_y)
        #             for i in range(4):
        #                 if lines[-(i + 1)].find('object:' + str(track_finished['id'])) == -1:
        #                     trace_cen_x = int((bbox[0] + bbox[2]) / 2) + move_img[frame_num - (i + 1)][0]
        #                     trace_cen_y = int((bbox[1] + bbox[3]) / 2) + move_img[frame_num - (i + 1)][1]
        #                     lines[-(i + 1)] = lines[-(i + 1)] + ' ' + 'object:' + str(track_finished['id']) + ' ' + str(trace_cen_x) + ' ' + str(trace_cen_y)
        #                     lines[-(i + 1)] = lines[-(i + 1)][:10] + str(int(lines[-(i + 1)][10]) + 1) + lines[-(i + 1)][11:]
        #     lines.append(line)
        #     frame_num += 1
        # with open(os.path.join('/data/result', folder + '.txt'), 'w') as f:
        #     f.write('铿锵四人行' + ' ' + str(folder) + ' ' + str(len(images)) + ' ' + str(id) + '\n')
        #     for line in lines:
        #         f.write(line + '\n')


if __name__ == '__main__':
    main()
