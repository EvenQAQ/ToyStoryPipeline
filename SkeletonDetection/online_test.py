import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.compat.v1 import ConfigProto
from utils import detector_utils as detector_utils
import cv2
import tensorflow as tf
import multiprocessing
from multiprocessing import Queue, Pool
import time
from utils.detector_utils import WebcamVideoStream
import datetime
import argparse
from sys import platform
import os
import sys
# First import the library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2
import pyrealsense2 as rs
# from LucasKanade import *
# from TemplateCorrection import *
from utils.imageProcess import *
frame_processed = 0
score_thresh = 0.2


tf_config = ConfigProto()
tf_config.gpu_options.allow_growth = True
session = InteractiveSession(config=tf_config)


def debug(string):
    print('\n')
    print(string)
    print('\n')

# Create a worker thread that loads graph and
# does detection on images in an input queue and puts it on an output queue




script_dir = 'D:/DevTools/openpose/build/examples/tutorial_api_python'

lines_hardcode = [[15,0], [16,0], [0,1],[1,2],[2,3],[3,4],[1,5],[5,6],[6,7],[1,8],[8,9],[9,10],[10,11],[11,22],[8,12],[12,13],[13,14],[14,19]]
data = np.array([np.array([[0,0.1],[0,0.1],[0,0.1]]) for line in lines_hardcode])
httpd = None
pose3ds = [[0,0,0] for i in range(25)]
#pose3ds = [[0,0,0] for i in range(25)]

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
daemon.setDaemon(True) # Set as a daemon so it will be killed once the main thread is dead.
daemon.start()


def update_lines(num, t, lines):
    global data
    for i in range(len(lines)):
        lines[i].set_data(data[i][0:2, :2])
        lines[i].set_3d_properties(data[i][2, :2])
    return lines

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-src',
        '--source',
        dest='video_source',
        type=int,
        default=1,
        help='Device index of the camera.')
    parser.add_argument(
        '-nhands',
        '--num_hands',
        dest='num_hands',
        type=int,
        default=2,
        help='Max number of hands to detect.')
    parser.add_argument(
        '-fps',
        '--fps',
        dest='fps',
        type=int,
        default=1,
        help='Show FPS on detection/display visualization')
    parser.add_argument(
        '-wd',
        '--width',
        dest='width',
        type=int,
        default=300,
        help='Width of the frames in the video stream.')
    parser.add_argument(
        '-ht',
        '--height',
        dest='height',
        type=int,
        default=200,
        help='Height of the frames in the video stream.')
    parser.add_argument(
        '-ds',
        '--display',
        dest='display',
        type=int,
        default=1,
        help='Display the detected images using OpenCV. This reduces FPS')
    parser.add_argument(
        '-num-w',
        '--num-workers',
        dest='num_workers',
        type=int,
        default=4,
        help='Number of workers.')
    parser.add_argument(
        '-q-size',
        '--queue-size',
        dest='queue_size',
        type=int,
        default=5,
        help='Size of the queue.')
    args = parser.parse_args()

    debug('parser')
    input_q = Queue(maxsize=args.queue_size)
    output_q = Queue(maxsize=args.queue_size)

    #video_capture = WebcamVideoStream(
    #    src=args.video_source, width=args.width, height=args.height).start()

    cap_params = {}
    frame_processed = 0
    cap_params['im_width'], cap_params['im_height'] = 640, 480
    cap_params['score_thresh'] = score_thresh

    # max number of hands we want to detect/track
    cap_params['num_hands_detect'] = args.num_hands


    print(cap_params, args)

    debug('capinfo')

    # spin up workers to paralleize detection.
    pool = Pool(args.num_workers, worker,
                (input_q, output_q, cap_params, frame_processed))

    start_time = datetime.datetime.now()
    num_frames = 0
    fps = 0
    index = 0


    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)


    # Start streaming
    pipe_profile = pipeline.start(config)
    profile = pipe_profile.get_stream(rs.stream.depth)
    depth_intrin = profile.as_video_stream_profile().get_intrinsics()
    print(depth_intrin)
    depth_sensor = pipe_profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    clip_long = 600
    clip_short = 150
    image_max = 255
    #openpose
    try:
        # Import Openpose (Windows/Ubuntu/OSX)
        # dir_path = os.path.dirname(os.path.realpath(__file__))
        dir_path = script_dir
        try:
            # Windows Import
            if platform == "win32":
                # Change these variables to point to the correct folder (Release/x64 etc.)
                sys.path.append(dir_path + '/../../python/openpose/Release');
                os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' +  dir_path + '/../../bin;'
                import pyopenpose as op
            else:
                # Change these variables to point to the correct folder (Release/x64 etc.)
                sys.path.append(dir_path + '../../python')
                # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
                # sys.path.append('/usr/local/python')
                from openpose import pyopenpose as op
        except ImportError as e:
            print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
            raise e

        # Flags

        parser = argparse.ArgumentParser()
        parser.add_argument("--image_path", default=dir_path + "../../../examples/media/COCO_val2014_000000000192.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
        args_openpose = parser.parse_known_args()

        # Custom Params (refer to include/openpose/flags.hpp for more parameters)
        params = dict()
        params["model_folder"] = dir_path + "/../../../models/"
        #params["tracking"] = 5
        params["hand"] = False
        params["hand_detector"] = 0
        params["body"] = 1
        #params["write_json"] = True



        # Add others in path?
        for i in range(0, len(args_openpose[1])):
            curr_item = args_openpose[1][i]
            if i != len(args_openpose[1])-1: next_item = args_openpose[1][i+1]
            else: next_item = "1"
            if "--" in curr_item and "--" in next_item:
                key = curr_item.replace('-','')
                if key not in params:  params[key] = "1"
            elif "--" in curr_item and "--" not in next_item:
                key = curr_item.replace('-','')
                if key not in params: params[key] = next_item

        # Construct it from system arguments
        # op.init_argv(args[1])
        # oppython = op.OpenposePython()

        debug("start op")
        # Starting OpenPose
        opWrapper = op.WrapperPython()
        opWrapper.configure(params)
        opWrapper.start()
        align_to = rs.stream.color
        align = rs.align(align_to)

        t = 0
        # Creating the Animation object
        #plt.ion()

        #plt.show()
        while True:
            debug('new frame')
            datum = op.Datum()
            # Get frameset of color and depth
            frames = pipeline.wait_for_frames()
            # frames.get_depth_frame() is a 640x360 depth image

            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()

            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue

            depth_origin = aligned_depth_frame.get_data()
            depth_image = np.asanyarray(depth_origin)
            color_origin = color_frame.get_data()
            color_image = np.asanyarray(color_origin)

            clipping_short = 150

            # Remove background - Set pixels further than clip_long to grey
            grey_color = 153
            depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
            bg_removed = np.where((depth_image_3d > clip_long) | (depth_image_3d <= clip_short), grey_color, color_image)
            depth_filtered = np.where((depth_image_3d > clip_long) | (depth_image_3d <= clip_short), 0, (depth_image_3d-clip_short) / (clip_long-clip_short) * 255)
            input_q.put(cv2.cvtColor(bg_removed, cv2.COLOR_BGR2RGB))

            cv2.namedWindow('MultiWindow Display', cv2.WINDOW_AUTOSIZE)
            cvoutput = np.hstack((color_image, bg_removed))
            cv2.imshow('MultiWindow Display', cvoutput)

            output_frame, boxes= output_q.get()
            debug("output frame")
            debug(output_frame)
            output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
            boxes_hand = [] # box in [x1, y1, x2, y2] for LK-filter
            boxes_quli = []

            debug("before loop")
            for box in boxes:
                box_scaled = [box[1]*cap_params['im_width'], box[0]*cap_params['im_height'], box[3]*cap_params['im_width'], box[2]*cap_params['im_height']]
                if boxQualified(box_scaled):
                    boxes_quli.append(box)

            debug("boxes")

            boxes = boxes_quli
            imageToProcess = bg_removed
            debug("imageToProcess")
            debug(imageToProcess)
            if len(boxes) >= 2:
                boxes_hand = [[max(boxes[0][1]*cap_params['im_width'],0), max(boxes[0][0]*cap_params['im_height'],0), min(boxes[0][3]*cap_params['im_width'],cap_params['im_width']-1), min(boxes[0][2]*cap_params['im_height'],cap_params['im_height']-1)],[max(boxes[1][1]*cap_params['im_width'],0), max(boxes[1][0]*cap_params['im_height'],0), min(boxes[1][3]*cap_params['im_width'],cap_params['im_width']-1), min(boxes[1][2]*cap_params['im_height'],cap_params['im_height']-1)]]
                box_left_final = boxes_hand[0]
                imageToProcess = cv2.rectangle(imageToProcess, (int(box_left_final[0]),int(box_left_final[1])), (int(box_left_final[2]), int(box_left_final[3])), (153,153,153),-1)
                box_left_final = boxes_hand[1]
                imageToProcess = cv2.rectangle(imageToProcess, (int(box_left_final[0]), int(box_left_final[1])), (int(box_left_final[2]), int(box_left_final[3])), (153, 153, 153), -1)
            debug("imageToProcess")
            debug(imageToProcess)
            datum.cvInputData = imageToProcess
            opWrapper.emplaceAndPop([datum])
            image_out = datum.cvOutputData
            debug("image_out")
            debug(image_out)
            # debug("hstack")
            # cvoutput = np.hstack((color_image, datum.cvOutputData))
            pose_keypoints = datum.poseKeypoints
            if pose_keypoints.size > 1:
                pose3ds = threeFromDepth(depth_filtered, depth_scale, clip_long, clip_short, image_max, depth_intrin, pose_keypoints[0])
            cv2.namedWindow('Image', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Image', image_out)
            key = cv2.waitKey(15)
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
    except Exception as e:
        print(e)
        sys.exit(-1)
