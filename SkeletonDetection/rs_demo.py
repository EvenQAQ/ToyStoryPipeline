from argparse import ArgumentParser
import json
import os

import cv2
import numpy as np
import datetime
import time
import pyrealsense2 as rs
import multiprocessing
from multiprocessing import Queue, Pool

from modules.input_reader import VideoReader, ImageReader
from modules.draw import Plotter3d, draw_poses
from modules.parse_poses import parse_poses

# from LucasKanade import *
# from TemplateCorrection import *
# from utils.imageProcess import *·

from http.server import HTTPServer, BaseHTTPRequestHandler
import threading

# lines_hardcode = [[15,0], [16,0], [0,1],[1,2],[2,3],[3,4],[1,5],[5,6],[6,7],[1,8],[8,9],[9,10],[10,11],[11,22],[8,12],[12,13],[13,14],[14,19]]
lines_hardcode = [[0, 1], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [
    6, 7], [1, 8], [1, 11], [8, 9], [9, 10], [11, 12], [12, 13]]
data = np.array([np.array([[0, 0.1], [0, 0.1], [0, 0.1]])
                 for line in lines_hardcode])
httpd = None
pose3ds = [[0, 0, 0] for i in range(18)]

coco2body25 = [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 9], [
    9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18]]


def pose_model_trans(rules, pose_to_trans):
    ans = [[0, 0, 0] for i in range(25)]
    for i in rules:
        ans[i[1]] = pose_to_trans[i[0]]
    return ans


def threeFromDepth(figure_depth, depth_scale, clip_long, clip_short, image_max, depth_intrin, points_2d):
    array_fig = np.asanyarray(figure_depth)
    num_row = len(array_fig)
    num_col = len(array_fig[0])
    points = []
    for i in range(len(points_2d)):
        depth = array_fig[int(points_2d[i][0])][int(points_2d[i][2])][0]
        # print (depth)
        # if depth < 0.1:
        #     points.append([-1, -1, -1])
        # else:
        #     depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, [int(points_2d[i][0]), int(
        #         points_2d[i][2])], (depth/image_max*(clip_long-clip_short) + clip_short)*depth_scale)
        #     points.append(depth_point)
        depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, [int(points_2d[i][0]), int(
            points_2d[i][2])], (depth/image_max*(clip_long-clip_short) + clip_short)*depth_scale)
        points.append(depth_point)
    return points


def rotate_poses(poses_3d, R, t):
    R_inv = np.linalg.inv(R)
    for pose_id in range(len(poses_3d)):
        pose_3d = poses_3d[pose_id].reshape((-1, 4)).transpose()
        pose_3d[0:3, :] = np.dot(R_inv, pose_3d[0:3, :] - t)
        poses_3d[pose_id] = pose_3d.transpose().reshape(-1)
    return poses_3d


def update_lines(num, t, lines):
    global data
    for i in range(len(lines)):
        lines[i].set_data(data[i][0:2, :2])
        lines[i].set_3d_properties(data[i][2, :2])
    return lines


def start_server():
    global httpd
    httpd = HTTPServer(('127.0.0.1', 8000), SimpleHTTPRequestHandler)
    httpd.serve_forever()


class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        global pose3ds
        content = ""
        for point in pose3ds:
            for value in point:
                content += str(value) + ' '
        self.wfile.write(content.encode('utf-8'))


daemon = threading.Thread(name='daemon_server',
                          target=start_server,
                          args=())
# Set as a daemon so it will be killed once the main thread is dead.
daemon.setDaemon(True)
daemon.start()


if __name__ == '__main__':
    parser = ArgumentParser(description='Lightweight 3D human pose estimation demo. '
                                        'Press esc to exit, "p" to (un)pause video or process next image.')
    parser.add_argument('-m', '--model',
                        help='Required. Path to checkpoint with a trained model '
                             '(or an .xml file in case of OpenVINO inference).',
                        type=str, required=True)
    parser.add_argument(
        '--video', help='Optional. Path to video file or camera id.', type=str, default='')
    parser.add_argument('-d', '--device',
                        help='Optional. Specify the target device to infer on: CPU or GPU. '
                             'The demo will look for a suitable plugin for device specified '
                             '(by default, it is GPU).',
                        type=str, default='GPU')
    parser.add_argument('--use-openvino',
                        help='Optional. Run network with OpenVINO as inference engine. '
                             'CPU, GPU, FPGA, HDDL or MYRIAD devices are supported.',
                        action='store_true')
    parser.add_argument(
        '--images', help='Optional. Path to input image(s).', nargs='+', default='')
    parser.add_argument(
        '--height-size', help='Optional. Network input layer height size.', type=int, default=256)
    parser.add_argument('--extrinsics-path',
                        help='Optional. Path to file with camera extrinsics.',
                        type=str, default=None)
    parser.add_argument('--fx', type=np.float32, default=-1,
                        help='Optional. Camera focal length.')
    args = parser.parse_args()

    if args.video == '' and args.images == '':
        raise ValueError('Either --video or --image has to be provided')

    stride = 8
    if args.use_openvino:
        from modules.inference_engine_openvino import InferenceEngineOpenVINO
        net = InferenceEngineOpenVINO(args.model, args.device)
    else:
        from modules.inference_engine_pytorch import InferenceEnginePyTorch
        net = InferenceEnginePyTorch(args.model, args.device)

    canvas_3d = np.zeros((480, 640, 3), dtype=np.uint8)
    plotter = Plotter3d(canvas_3d.shape[:2])
    canvas_3d_window_name = 'Canvas 3D'
    cv2.namedWindow(canvas_3d_window_name)
    cv2.setMouseCallback(canvas_3d_window_name, Plotter3d.mouse_callback)

    file_path = args.extrinsics_path
    if file_path is None:
        file_path = os.path.join('data', 'extrinsics.json')
    with open(file_path, 'r') as f:
        extrinsics = json.load(f)
    R = np.array(extrinsics['R'], dtype=np.float32)
    t = np.array(extrinsics['t'], dtype=np.float32)

    frame_provider = ImageReader(args.images)
    is_video = False
    if args.video != '':
        frame_provider = VideoReader(args.video)
        is_video = True
    base_height = args.height_size
    fx = args.fx

    delay = 1
    esc_code = 27
    p_code = 112
    space_code = 32
    mean_time = 0

    # set realsense
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
    # Start streaming
    profile = pipeline.start(config)
    depth_profile = rs.video_stream_profile(
        profile.get_stream(rs.stream.depth))
    depth_intrin = depth_profile.get_intrinsics()
    # print(depth_intrin)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: ", depth_scale)
    clip_long_in_meters = 0.5
    clip_short_in_meters = 0.05
    clip_long = clip_long_in_meters / depth_scale
    clip_short = clip_short_in_meters / depth_scale
    image_max = 255

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    frame_gap = 3
    frame_recount = frame_gap * 100
    start_time = datetime.datetime.now()
    num_frames = 0
    num_addframes = 0
    fps = 0

    while True:
        # current_time = cv2.getTickCount()
        # count frames
        if (num_frames == frame_recount):
            print("frame recount by" + str(frame_recount))
            print("add frames" + str(num_addframes))
            num_addframes = 0
            num_frames = 0
            start_time = datetime.datetime.now()
        # if (num_frames == 60):
        #     print("clean frames num")
        #     num_frames = 0
        #     start_time = datetime.datetime.now()
        num_frames += 1
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        # aligned_depth_frame is a 640x480 depth image
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        depth_origin = aligned_depth_frame.get_data()
        depth_image = np.asanyarray(depth_origin)
        color_origin = color_frame.get_data()
        color_image = np.asanyarray(color_origin)

        # Remove background - Set pixels further than clipping_distance to grey
        grey_color = 153

        # depth image is 1 channel, color is 3 channels
        depth_image_3d = np.dstack(
            (depth_image, depth_image, depth_image))
        bg_removed = np.where((depth_image_3d > clip_long) | (
            depth_image_3d <= clip_short), grey_color, color_image)
        depth_filtered = np.where((depth_image_3d > clip_long) | (
            depth_image_3d <= clip_short), 0, (depth_image_3d-clip_short) / (clip_long-clip_short) * 255)

        frame = bg_removed

        # frame = color_image
        if frame is None:
            break
        input_scale = base_height / frame.shape[0]
        scaled_img = cv2.resize(
            frame, dsize=None, fx=input_scale, fy=input_scale)
        # better to pad, but cut out for demo
        scaled_img = scaled_img[:, 0:scaled_img.shape[1] -
                                (scaled_img.shape[1] % stride)]
        if fx < 0:  # Focal length is unknown
            fx = np.float32(0.8 * frame.shape[1])

        inference_result = net.infer(scaled_img)
        poses_3d, poses_2d = parse_poses(
            inference_result, input_scale, stride, fx, is_video)
        # print(poses_2d)
        # print(poses_3d)
        # print ("line")

        edges = []
        if len(poses_3d):
            poses_3d = rotate_poses(poses_3d, R, t)
            poses_3d_copy = poses_3d.copy()
            x = poses_3d_copy[:, 0::4]
            y = poses_3d_copy[:, 1::4]
            z = poses_3d_copy[:, 2::4]
            poses_3d[:, 0::4], poses_3d[:, 1::4], poses_3d[:, 2::4] = -z, x, -y

            poses_3d = poses_3d.reshape(poses_3d.shape[0], 19, -1)[:, :, 0:3]
            edges = (Plotter3d.SKELETON_EDGES + 19 *
                     np.arange(poses_3d.shape[0]).reshape((-1, 1, 1))).reshape((-1, 2))
        # if poses_3d.size > 1:
        #     print('\r' + str(poses_3d[0][7]))
        if poses_3d.size > 1:
            # pose3ds = poses_3d[0]
            pose3ds = threeFromDepth(
                depth_filtered, depth_scale, clip_long, clip_short, image_max, depth_intrin, poses_3d[0])
        print(pose3ds)
        plotter.plot(canvas_3d, poses_3d, edges)
        cv2.imshow(canvas_3d_window_name, canvas_3d)

        draw_poses(frame, poses_2d)

        # storage a frame for add
        storage_frame = frame

        # show fps
        elapsed_time = (datetime.datetime.now() -
                        start_time).total_seconds()
        fps = num_frames / elapsed_time
        cv2.putText(frame, 'FPS: {}'.format(fps),
                    (40, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))

        cv2.imshow('ICV 3D Human Pose Estimation', frame)

        key = cv2.waitKey(delay)
        if key == esc_code:
            break
        if key == p_code:
            if delay == 1:
                delay = 0
            else:
                delay = 1
        if delay == 0 or not is_video:  # allow to rotate 3D canvas while on pause
            key = 0
            while (key != p_code
                   and key != esc_code
                   and key != space_code):
                plotter.plot(canvas_3d, poses_3d, edges)
                cv2.imshow(canvas_3d_window_name, canvas_3d)
                key = cv2.waitKey(33)
            if key == esc_code:
                break
            else:
                delay = 1

        # add frame

        if (num_frames % frame_gap == 0):
            num_frames += 1
            num_addframes += 1

            cv2.imshow('ICV 3D Human Pose Estimation', frame)
            key = cv2.waitKey(delay)
            if key == esc_code:
                break
            if key == p_code:
                if delay == 1:
                    delay = 0
                else:
                    delay = 1
            if delay == 0 or not is_video:  # allow to rotate 3D canvas while on pause
                key = 0
                while (key != p_code
                       and key != esc_code
                       and key != space_code):
                    plotter.plot(canvas_3d, poses_3d, edges)
                    cv2.imshow(canvas_3d_window_name, canvas_3d)
                    key = cv2.waitKey(33)
                if key == esc_code:
                    break
                else:
                    delay = 1
