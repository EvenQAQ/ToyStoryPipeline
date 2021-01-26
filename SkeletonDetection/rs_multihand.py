# coding=utf-8

import pyrealsense2 as rs   # First import library
import numpy as np          # Import Numpy for easy array manipulation
import cv2                  # Import OpenCV for easy image rendering
import argparse             # Import argparse for command-line options
import os.path              # Import os.path for file path manipulation
import os
import sys
from sys import platform
import time
import datetime
import tensorflow as tf
from utils import detector_utils


script_dir = 'D://Codes/openpose/build/examples/tutorial_api_python'

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
        # Create object for parsing command-line options
        parser = argparse.ArgumentParser(description="Read recorded bag file and display depth stream in jet colormap.\
                                        Remember to change the stream resolution, fps and format to match the recorded.")
        # Add argument which takes path to a bag file as an input
        parser.add_argument("-i", "--input", type=str,
                            help="Path to the bag file")
        # parser.add_argument("-o", "--output", type=str, help = "Path to save the output")
        # Parse the command line arguments to an object
        args = parser.parse_args()
        # Safety if no parameter have been given
        if not args.input:
            print("No input paramater have been given.")
            print("For help type --help")
            exit()
        # Check if the given file have bag extension
        if os.path.splitext(args.input)[1] != ".bag":
            print("The given file is not of correct file format.")
            print("Only .bag files are accepted")
            exit()
        # if not args.output:
        #     print("No output paramater have been given.")
        #     print("For help type --help")
        #     exit()
        # else:
        #     output_file = args.output
        # save_fps = 60
        # fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', '2')
        # videoWriter = cv2.VideoWriter(
        #     output_file, fourcc, save_fps, (640, 480))  # 最后一个是保存图片的尺寸
        # openpose configuration
        params = dict()
        params["model_folder"] = dir_path + "/../../../models/"
        params["hand"] = True
        params["hand_detector"] = 2
        params["body"] = 0

        opWrapper = op.WrapperPython()
        opWrapper.configure(params)
        opWrapper.start()

        # setting hand bounding box inference
        detection_graph, sess = detector_utils.load_inference_graph()
        sess = tf.compat.v1.Session(graph=detection_graph)

        try:
            # Create pipeline
            pipeline = rs.pipeline()

            # Create a config object
            config = rs.config()
            # Tell config that we will use a recorded device from filem to be used by the pipeline through playback.
            rs.config.enable_device_from_file(config, args.input)
            # Configure the pipeline to stream the depth stream
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 90)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
            # Start streaming from file
            pipeline.start(config)

            # Create opencv window to render image in
            cv2.namedWindow("handtrack by openpose", cv2.WINDOW_AUTOSIZE)

            # Create colorizer object
            colorizer = rs.colorizer()

            # Streaming loop
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
                color_image = np.asanyarray(color_frame.get_data())

                # Create new datum
                datum = op.Datum()
                datum.cvInputData = color_image

                # draw bounding box
                num_hands_detect = 2
                boxes, scores = detector_utils.detect_objects(
                    color_image, detection_graph, sess)
                handRectangles = [
                    [op.Rectangle(0., 0., 0., 0.)for i in range(2)]]
                im_height = 480
                im_width = 640
                score_thresh = 0.2
                for i in range(num_hands_detect):
                    if (scores[i] > score_thresh):
                        (left, right, top, bottom) = (
                            boxes[i][1] * im_width, boxes[i][3] * im_width, boxes[i][0] * im_height, boxes[i][2] * im_height)
                        rec_width = right - left
                        rec_height = bottom - top
                        handRectangles[0][i] = op.Rectangle(
                            left, top, rec_width, rec_height)

                # op.Rectangle(lefttop_x,lefttop_y,width,height)
                print("rec start")
                print(handRectangles)
                print("rec end")

                datum.handRectangles = handRectangles
                opWrapper.emplaceAndPop([datum])

                # cvoutput=np.hstack((depth_color_image, color_image))
                cvoutput = datum.cvOutputData
                # Render image in opencv window
                # videoWriter.write(cvoutput)
                cv2.imshow("handtrack by openpose", cvoutput)
                key = cv2.waitKey(1)
                # if pressed escape exit program
                if key == 27:
                    cv2.destroyAllWindows()
                    break
        finally:
            pass
    except Exception as e:
        print(e)
        sys.exit(-1)
