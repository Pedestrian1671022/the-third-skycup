import os
import cv2
import glob
import numpy as np
import tensorflow as tf
import core.utils as utils


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

def track_iou(detections, tracks_active, sigma_l, sigma_h, sigma_iou, t_life, t_loss, id):
    tracks_finished = []
    dets = [det for det in detections if det['score'] >= sigma_l]
    updated_tracks = []
    for track in tracks_active:
        if len(dets) > 0:
            best_match = max(dets, key=lambda x: iou(track['bboxes'], x['bbox']))
            if iou(track['bboxes'], best_match['bbox']) >= sigma_iou:
                track['bboxes'] = best_match['bbox']
                track['max_score'] = max(track['max_score'], best_match['score'])
                track['life_time'] += 1
                track['loss_time'] = 0
                updated_tracks.append(track)
                del dets[dets.index(best_match)]
                if track['max_score'] >= sigma_h and track['life_time'] >= t_life:
                    if track['id'] == 0:
                        id += 1
                        track['id'] = id
                    tracks_finished.append(track)
        if len(updated_tracks) == 0 or track is not updated_tracks[-1]:
            track['loss_time'] += 1
            if track['loss_time'] > 10:
                track['life_time'] = 0
            if track['loss_time'] >= t_loss:
                tracks_active.remove(track)
    new_tracks = [{'bboxes': det['bbox'], 'max_score': det['score'], 'life_time':1, 'loss_time': 0, 'id': 0} for det in dets]
    tracks_active = tracks_active + new_tracks
    return tracks_finished, tracks_active, id

def convertBack(xmin, ymin, xmax, ymax, conf):
    xmin = int(xmin)
    xmax = int(xmax)
    ymin = int(ymin)
    ymax = int(ymax)
    return [xmin, ymin, xmax, ymax], conf

return_elements = ['input/input_data:0', 'pred_sbbox/concat_2:0', 'pred_mbbox/concat_2:0', 'pred_lbbox/concat_2:0']
pb_file = 'yolov3.pb'
graph = tf.compat.v1.Graph()
return_tensors = utils.read_pb_return_tensors(graph, pb_file, return_elements)

num_classes = 1
input_size = 640

with tf.compat.v1.Session(graph=graph) as sess:
    folders = os.listdir('/data/input_data')
    for folder in folders:
        lines = []
        images = glob.glob(os.path.join('/data/input_data', folder, '*.bmp'))
        images = sorted(images)
        tracks_active = []
        id = 0
        for image in images:
            image_data = cv2.imread(image)
            image_size = image_data.shape[:2]
            input_data = utils.image_preporcess(image_data.copy(), [input_size, input_size])
            input_data = input_data[np.newaxis, ...]
            pred_sbbox, pred_mbbox, pred_lbbox = sess.run([return_tensors[1], return_tensors[2], return_tensors[3]], feed_dict={ return_tensors[0]: input_data})
            pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)), np.reshape(pred_mbbox, (-1, 5 + num_classes)), np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)
            bboxes = utils.postprocess_boxes(pred_bbox, image_size, input_size, 0.02)
            bboxes = utils.nms(bboxes, 0.45, method='nms')
            bboxes = [convertBack(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]), float(bbox[4])) for bbox in bboxes]
            detections = [{'bbox': bbox, 'score': conf} for bbox, conf in bboxes]
            tracks_finished, tracks_active, id = track_iou(detections, tracks_active, 0.05, 0.15, 0.1, 5, 15, id)
            for track_finished in tracks_finished:
                bbox = track_finished['bboxes']
                cv2.rectangle(image_data, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
                cv2.putText(image_data, str(track_finished['id']), (int(bbox[0]), int(bbox[3])), 0, 5e-3 * 130, (0, 255, 0),2)
            cv2.imshow('', image_data)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        #     if len(tracks_finished) == 0:
        #         line = 'frame:' + image[-7:-4] + ' ' + str(len(tracks_finished))
        #     else:
        #         line = 'frame:' + image[-7:-4] + ' ' + str(len(tracks_finished))
        #         for track_finished in tracks_finished:
        #             bbox = track_finished['bboxes']
        #             cen_x = int((bbox[0] + bbox[2]) / 2)
        #             cen_y = int(((bbox[1] + bbox[3]) / 2))
        #             line = line + ' ' + 'object:' + str(track_finished['id']) + ' ' + str(cen_x) + ' ' + str(cen_y)
        #             for i in range(4):
        #                 if lines[-(i+1)].find('object:' + str(track_finished['id'])) == -1:
        #                     lines[-(i+1)] = lines[-(i+1)] + ' ' + 'object:' + str(track_finished['id']) + ' ' + str(cen_x) + ' ' + str(cen_y)
        #                     lines[-(i+1)] = lines[-(i+1)][:10] + str(int(lines[-(i+1)][10]) + 1) + lines[-(i+1)][11:]
        #         line = line
        #     lines.append(line)
        # with open(os.path.join('/data/result', folder + '.txt'), 'w') as f:
        #     f.write('铿锵四人行' + ' ' + str(folder) + ' ' + str(len(images)) + ' ' + str(id) + '\n')
        #     for line in lines:
        #         f.write(line + '\n')
