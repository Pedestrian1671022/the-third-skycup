import os
import cv2
import glob
import numpy as np
import tensorflow as tf
import core.utils as utils
from tensorflow.python.saved_model import tag_constants


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
        if len(dets) > 0:
            # get det with highest iou
            best_match = max(dets, key=lambda x: iou(track['bboxes'], x['bbox']))
            if iou(track['bboxes'], best_match['bbox']) >= sigma_iou:
                track['bboxes'] = best_match['bbox']
                track['max_score'] = max(track['max_score'], best_match['score'])
                track['life_time'] += 1
                track['loss_time'] = 0
                updated_tracks.append(track)
                del_dets.append(dets[dets.index(best_match)])
                # remove from best matching detection from detections
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
                   'max_score': det['score'], 'life_time':1, 'loss_time': 0, 'id': 0} for det in dets]
    tracks_active = tracks_active + new_tracks

    return tracks_finished, tracks_active, tracks_loss, id

def convertBack(xmin, ymin, xmax, ymax, conf):
    xmin = int(xmin)
    xmax = int(xmax)
    ymin = int(ymin)
    ymax = int(ymax)
    return [xmin, ymin, xmax, ymax], conf

saved_model_loaded = tf.saved_model.load('./yolov4-640/', tags=[tag_constants.SERVING])
infer = saved_model_loaded.signatures['serving_default']
feature_params = dict(maxCorners=20, qualityLevel=0.3, minDistance=7, blockSize=21)
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

num_classes = 1
input_size = 640


folders = os.listdir('/data/input_data')
for folder in folders:
    # lines = []
    images = glob.glob(os.path.join('/data/input_data', folder, '*.bmp'))
    images = sorted(images)
    tracks_active = []
    tracks_loss = []
    id = 0
    lines = []
    frame_num = 0
    move = np.array([0, 0])
    move_img = np.zeros((250, 2), dtype=np.int)

    for image in images:
        image_data = cv2.imread(image)
        image_size = image_data.shape[:2]
        input_data = cv2.resize(image_data, (input_size, input_size))
        input_data = input_data / 255.

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

        input_data = tf.constant(input_data[np.newaxis, ...].astype(np.float32))
        pred_bbox = infer(input_data)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            confs = value[:, :, 4:]
        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(confs, (tf.shape(confs)[0], -1, tf.shape(confs)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=0.45,
            score_threshold=0.1
        )
        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]

        bboxes = boxes.numpy()[0]

        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]
        # print(num_objects, bboxes, scores, classes)

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = image_data.shape
        # bboxes,=truth_bbox(image[-7:-4], 30)
        bboxes = utils.format_boxes(bboxes, original_h, original_w, move)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        # read in all class names from config
        class_names = {0:'car'}

        # by default allow all classes in .names file
        allowed_classes = list(class_names.values())

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        count = len(names)
        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        detections = [{'bbox': bbox.tolist(), 'score': conf} for bbox, conf in zip(bboxes, scores)]
        tracks_finished, tracks_active, tracks_loss, id = track_iou(detections, tracks_active, tracks_loss, 0.1, 0.2, 0.01, 10, 20, id, int(image[-7:-4]))
        for track_finished in tracks_finished:
            bbox = track_finished['bboxes']
            cv2.rectangle(image_data, (int(bbox[0]+move[0]), int(bbox[1]+move[1])), (int(bbox[2]+move[0]), int(bbox[3]+ move[1])), (255, 0, 0), 2)
            cv2.putText(image_data, str(track_finished['id']), (int(bbox[0]+move[0]), int(bbox[3]+move[1])), 0, 5e-3 * 130, (0, 255, 0),2)
        cv2.namedWindow("show", cv2.WINDOW_NORMAL)
        cv2.imshow('show', image_data)
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
    #             for i in range(9):
    #                 if lines[-(i + 1)].find('object:' + str(track_finished['id'])) == -1:
    #                     trace_cen_x = int((bbox[0] + bbox[2]) / 2) + move_img[frame_num - (i + 1)][0]
    #                     trace_cen_y = int((bbox[1] + bbox[3]) / 2) + move_img[frame_num - (i + 1)][1]
    #                     lines[-(i + 1)] = lines[-(i + 1)] + ' ' + 'object:' + str(track_finished['id']) + ' ' + str(
    #                         trace_cen_x) + ' ' + str(trace_cen_y)
    #                     lines[-(i + 1)] = lines[-(i + 1)][:10] + str(int(lines[-(i + 1)][10]) + 1) + lines[
    #                                                                                                      -(i + 1)][
    #                                                                                                  11:]
    #     lines.append(line)
    #     frame_num += 1
    # with open(os.path.join('/data/result', folder + '.txt'), 'w') as f:
    #     f.write('铿锵四人行' + ' ' + str(folder) + ' ' + str(len(images)) + ' ' + str(id) + '\n')
    #     for line in lines:
    #         f.write(line + '\n')
