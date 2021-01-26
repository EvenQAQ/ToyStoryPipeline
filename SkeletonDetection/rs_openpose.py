#coding=utf-8
# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
import pyrealsense2 as rs
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import datetime

script_dir = 'D://Codes/openpose/build/examples/tutorial_api_python'

lines_hardcode = [[15,0], [16,0], [0,1],[1,2],[2,3],[3,4],[1,5],[5,6],[6,7],[1,8],[8,9],[9,10],[10,11],[11,22],[8,12],[12,13],[13,14],[14,19]]
data = np.array([np.array([[0,0.1],[0,0.1],[0,0.1]]) for line in lines_hardcode])
httpd = None
pose3ds = [[0, 0, 0] for i in range(25)]


def array2str(array):
    string = str(array)
    string = string.replace('[', '')
    string = string.replace(']', '')
    string = string.replace('\n', '')
    return string


def remove_background(depth_image, depth_scale, clipping_distance):
    return

def hand_filter():
    return

def optical_flow():
    return

def kalman_filter():
    return

def debug(string):
    print('\n')
    print(string)
    print('\n')

def draw_fps_on_image(fps, image_np):
    cv2.putText(image_np, fps, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)

def update_lines(num, t, lines):
    global data
    for i in range(len(lines)):
        lines[i].set_data(data[i][0:2, :2])
        lines[i].set_3d_properties(data[i][2, :2])
    return lines

if __name__ == "__main__":
    # FPS record
    start_time = datetime.datetime.now()
    num_frames = 0
    fps = 0
    index = 0
    cv2.namedWindow('MultiWindow Display', cv2.WINDOW_AUTOSIZE)
    # realsense setup
    # Process Image
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 90)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
    # Start streaming
    profile = pipeline.start(config)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    # depth_intrin = profile.video_stream_profile().get_intrinsics()
    # print(depth_intrin)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: ", depth_scale)
    clip_long = 600
    clip_short = 50
    image_max = 255

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    try:
        # Import Openpose (Windows/Ubuntu/OSX)
        dir_path = script_dir
        # dir_path = os.path.dirname(os.path.realpath(__file__))
        try:
            # Windows Import
            if platform == "win32":
                # Change these variables to point to the correct folder (Release/x64 etc.)
                sys.path.append(dir_path + '/../../python/openpose/Release');
                os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' +  dir_path + '/../../bin;'
                import pyopenpose as op
            else:
                # Change these variables to point to the correct folder (Release/x64 etc.)
                sys.path.append(dir_path + '/../../python');
                # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
                # sys.path.append('/usr/local/python')
                from openpose import pyopenpose as op
        except ImportError as e:
            print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
            raise e

        # Flags
        parser = argparse.ArgumentParser()
        parser.add_argument("--image_path", default=dir_path + "/../../../examples/media/COCO_val2014_000000000192.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
        args = parser.parse_known_args()

        # Custom Params (refer to include/openpose/flags.hpp for more parameters)
        params = dict()
        params["model_folder"] = dir_path + "/../../../models/"
        params["hand"] = True
        params["hand_detector"] = 0
        params["body"] = 1

        # Add others in path?
        for i in range(0, len(args[1])):
            curr_item = args[1][i]
            if i != len(args[1])-1: next_item = args[1][i+1]
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

        # Starting OpenPose
        opWrapper = op.WrapperPython()
        opWrapper.configure(params)
        opWrapper.start()


        try:
            while True:
                datum = op.Datum()

                # Get frameset of color and depth
                frames = pipeline.wait_for_frames()
                # frames.get_depth_frame() is a 640x360 depth image

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
                # depth image is 1 channel, color is 3 channels
                depth_image_3d = np.dstack(
                    (depth_image, depth_image, depth_image))
                bg_removed = np.where((depth_image_3d > clip_long) | (depth_image_3d <= clip_short), grey_color, color_image)
                depth_filtered = np.where((depth_image_3d > clip_long) | (depth_image_3d <= clip_short), 0, (depth_image_3d-clip_short) / (clip_long-clip_short) * 255)

                imageToProcess = bg_removed
                datum.cvInputData = imageToProcess
                opWrapper.emplaceAndPop([datum])

                # Display Image
                print('display')
                # print(datum.poseKeypoints)
                # if datum.poseKeypoints > 1:
                #     pose3ds = datum.poseKeypoints[0]
                # print(pose3ds)
                # print(type(datum.cvOutputData))

                cvoutput = np.hstack((color_image, datum.cvOutputData))
                elapsed_time = (datetime.datetime.now() -
                                start_time).total_seconds()
                num_frames += 1

                if num_frames < 50:
                    start_time = datetime.datetime.now()
                    continue
                fps = (num_frames - 49) / elapsed_time
                draw_fps_on_image(
                    "FPS : " + str(int(fps)), cvoutput)
                cv2.imshow('MultiWindow Display', cvoutput)
                key = cv2.waitKey(15)

                # Press esc or 'q' to close the image window
                if key & 0xFF == ord('q') or key == 27:
                    cv2.destroyAllWindows()
                    break
        finally:
            # Stop streaming
            pipeline.stop()


    except Exception as e:
        print(e)
        sys.exit(-1)
