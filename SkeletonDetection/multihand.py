# coding=utf-8
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

lines_hardcode = [[15, 0], [16, 0], [0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [
    6, 7], [1, 8], [8, 9], [9, 10], [10, 11], [11, 22], [8, 12], [12, 13], [13, 14], [14, 19]]
data = np.array([np.array([[0, 0.1], [0, 0.1], [0, 0.1]])
                 for line in lines_hardcode])
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
    try:
        # Import Openpose (Windows/Ubuntu/OSX)
        dir_path = script_dir
        # dir_path = os.path.dirname(os.path.realpath(__file__))
        try:
            # Windows Import
            if platform == "win32":
                # Change these variables to point to the correct folder (Release/x64 etc.)
                sys.path.append(dir_path + '/../../python/openpose/Release')
                os.environ['PATH'] = os.environ['PATH'] + ';' + dir_path + \
                    '/../../x64/Release;' + dir_path + '/../../bin;'
                import pyopenpose as op
            else:
                # Change these variables to point to the correct folder (Release/x64 etc.)
                sys.path.append(dir_path + '/../../python')
                # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
                # sys.path.append('/usr/local/python')
                from openpose import pyopenpose as op
        except ImportError as e:
            print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
            raise e
        parser = argparse.ArgumentParser(description="Read recorded bag file and display depth stream in jet colormap.\
                                    Remember to change the stream resolution, fps and format to match the recorded.")
        # Add argument which takes path to a bag file as an input
        # parser.add_argument("--image_path", default=dir_path + "/../../../examples/media/COCO_val2014_000000000192.jpg",
        #                     help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
        parser.add_argument("-i", "--input", type=str,
                            help="Path to the bag file")
        args = parser.parse_known_args()

        if not args.input:
            print("No input paramater have been given.")
            print("For help type --help")
            exit()
        # Check if the given file have bag extension
        if os.path.splitext(args.input)[1] != ".bag":
            print("The given file is not of correct file format.")
            print("Only .bag files are accepted")
            exit()

        # realsense setup
        # Process Image
        pipeline = rs.pipeline()
        config = rs.config()
        rs.config.enable_device_from_file(config, args.input)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 90)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)

        # Start streaming
        profile = pipeline.start(config)
        colorizer = rs.colorizer()
        # Getting the depth sensor's depth scale (see rs-align example for explanation)
        # depth_intrin = profile.video_stream_profile().get_intrinsics()
        # print(depth_intrin)
        # depth_sensor = profile.get_device().first_depth_sensor()
        # depth_scale = depth_sensor.get_depth_scale()
        # print("Depth Scale is: ", depth_scale)
        clip_long = 600
        clip_short = 50
        image_max = 255

        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
        align_to = rs.stream.color
        align = rs.align(align_to)
        # Custom Params (refer to include/openpose/flags.hpp for more parameters)
        params = dict()
        params["model_folder"] = dir_path + "/../../../models/"
        params["hand"] = True
        params["hand_detector"] = 2
        params["body"] = 0

        # Add others in path?
        # for i in range(0, len(args[1])):
        #     curr_item = args[1][i]
        #     if i != len(args[1])-1:
        #         next_item = args[1][i+1]
        #     else:
        #         next_item = "1"
        #     if "--" in curr_item and "--" in next_item:
        #         key = curr_item.replace('-', '')
        #         if key not in params:
        #             params[key] = "1"
        #     elif "--" in curr_item and "--" not in next_item:
        #         key = curr_item.replace('-', '')
        #         if key not in params:
        #             params[key] = next_item

        # Construct it from system arguments
        # op.init_argv(args[1])
        # oppython = op.OpenposePython()

        # Starting OpenPose
        opWrapper = op.WrapperPython()
        opWrapper.configure(params)
        opWrapper.start()

        # FPS record
        start_time = datetime.datetime.now()
        num_frames = 0
        fps = 0
        try:
            while True:
                # Get frameset of depth
                frames = pipeline.wait_for_frames()

                # Get depth frame
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()

                # Colorize depth frame to jet colormap
                depth_color_frame = colorizer.colorize(depth_frame)

                # Convert depth_frame to numpy array to render image in opencv
                depth_color_image = np.asanyarray(depth_color_frame.get_data())

                # Render image in opencv window
                cv2.imshow("Depth Stream", depth_color_image)
                key = cv2.waitKey(1)
                # if pressed escape exit program
                if key == 27:
                    cv2.destroyAllWindows()
                    break

                # cvoutput = np.hstack((color_image, datum.cvOutputData))
                # elapsed_time = (datetime.datetime.now() -
                #                 start_time).total_seconds()
                # num_frames += 1

                # if num_frames < 50:
                #     start_time = datetime.datetime.now()
                #     continue
                # fps = (num_frames - 49) / elapsed_time
                # draw_fps_on_image(
                #     "FPS : " + str(int(fps)), cvoutput)
                # cv2.imshow('MultiWindow Display', cvoutput)
                # key = cv2.waitKey(15)

                # # Press esc or 'q' to close the image window
                # if key & 0xFF == ord('q') or key == 27:
                #     cv2.destroyAllWindows()
                #     break
        finally:
            # Stop streaming
            pipeline.stop()

    except Exception as e:
        print(e)
        sys.exit(-1)
