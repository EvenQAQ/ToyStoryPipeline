import numpy as np
import sys
import tensorflow as tf
import os
from threading import Thread
import pyrealsense2 as rs
from utils import detector_utils as detector_utils
import cv2

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   try:
#     # Currently, memory growth needs to be the same across GPUs
#     for gpu in gpus:
#       tf.config.experimental.set_memory_growth(gpu, True)
#     logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#   except RuntimeError as e:
#     # Memory growth must be set before GPUs have been initialized
#     print(e)

def worker(input_q, output_q, cap_params, frame_processed):
    print(">> loading frozen model for worker")
    detection_graph, sess = detector_utils.load_inference_graph()
    sess = tf.Session(graph=detection_graph)
    while True:
        #print("> ===== in worker loop, frame ", frame_processed)
        frame = input_q.get()
        if (frame is not None):
            # Actual detection. Variable boxes contains the bounding box cordinates for hands detected,
            # while scores contains the confidence for each of these boxes.
            # Hint: If len(boxes) > 1 , you may assume you have found atleast one hand (within your score threshold)

            boxes, scores = detector_utils.detect_objects(
                frame, detection_graph, sess)

            # get qualified boexes
            new_boxes = []
            for i in range(len(boxes)):
                if scores[i] >= cap_params['score_thresh']:
                    new_boxes.append(boxes[i])
            # draw bounding boxes
            detector_utils.draw_box_on_image(
                cap_params['num_hands_detect'], cap_params["score_thresh"],
                scores, boxes, cap_params['im_width'], cap_params['im_height'],
                frame)
            # add frame annotated with bounding box to queue
            output_q.put([frame, new_boxes])
            frame_processed += 1
        else:
            output_q.put(frame)
    sess.close()

def pointQualified(point):
    threshold_point = 0.1
    return (point[0] > 0 and point[1] > 0)

def pointQualified3d(point):
    threshold_point = 0.1
    return (point[0] > -0.9 and point[1] > -0.9)


def boxQualified(rect):
    threshold_width = 100
    threshold_height = 100
    return (rect[2]-rect[0] >= threshold_width) and (rect[3] - rect[1] >= threshold_height)

def boxPointQualified(rect):
    threshold_width = 10
    threshold_height = 10
    return (rect[2]-rect[0] >= threshold_width) and (rect[3] - rect[1] >= threshold_height)

def threeFromXY(figure_depth, depth_scale, clip_long, clip_short, image_max, depth_intrin, points_2d):
    figure = cv2.imread(figure_depth)
    array_fig = np.asanyarray(figure)
    num_row = len(array_fig)
    num_col = len(array_fig[0])
    print(num_row)
    print(num_col)
    points = []
    for i in range(len(points_2d)):
        depth = array_fig[int(points_2d[i][1])][int(points_2d[i][0])][0]
        if depth < 0.1:
            points.append([-1,-1,-1])
        else:
            depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, [int(points_2d[i][1]),int(points_2d[i][0])], (depth/image_max*(clip_long-clip_short) + clip_short)*depth_scale)
            points.append(depth_point)
    return points

def threeFromDepth(figure_depth, depth_scale, clip_long, clip_short, image_max, depth_intrin, points_2d):
    array_fig = np.asanyarray(figure_depth)
    num_row = len(array_fig)
    num_col = len(array_fig[0])
    points = []
    for i in range(len(points_2d)):
        depth = array_fig[int(points_2d[i][0])][int(points_2d[i][1])][0]
        print (depth)
        if depth < 0.1:
            points.append([-1,-1,-1])
        else:
            depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, [int(points_2d[i][1]),int(points_2d[i][0])], (depth/image_max*(clip_long-clip_short) + clip_short)*depth_scale)
            points.append(depth_point)
    return points
