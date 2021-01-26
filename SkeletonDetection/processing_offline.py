# coding=utf-8

import sys
import cv2
import os
import platform
import argparse
import pyrealsense2 as rs
import numpy as np
import threading
import datetime

sysinfo = platform.system()
if sysinfo == 'Windows':
    script_dir = 'D://DevTools/openpose/build/examples/tutorial_api_python'
    # data_dir = 'D://Codes/ARTangibleAnimation/Data/Data'
elif sysinfo == 'Darwin':
    pass
    # data_dir = '/Users/evenqaq/Dev/Codes/ARTangibleAnimation/Data/Data'


def color_frame_name(num):
    return 'color' + str(num) + '.png'


def depth_frame_name(num):
    return 'depth' + str(num) + '.png'


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
        # Argument Parser
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '-src',
            '--source',
            dest='video_source',
            type=int,
            default=0,
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
            default=640,
            help='Width of the frames in the video stream.')
        parser.add_argument(
            '-ht',
            '--height',
            dest='height',
            type=int,
            default=480,
            help='Height of the frames in the video stream.')
        parser.add_argument(
            '-ds',
            '--display',
            dest='display',
            type=int,
            default=1,
            help='Display the detected images using OpenCV. This reduces FPS')
        parser.add_argument(
            '-i',
            '--input',
            dest='input_dir',
            type=str,
            default='D:\Codes\ToyStory\Data\Data',
            help='Data Directory')
        parser.add_argument(
            '-o',
            '--output',
            dest='output_dir',
            type=str,
            default='D:\Codes\ToyStory\Data\ProcessedData',
            help='output directory')
        parser.add_argument(
            "--image_path",
            default=dir_path + "/../../../examples/media/COCO_val2014_000000000192.jpg",
            help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
        args = parser.parse_known_args()
        print(args[0])
        video_source = args.video_source
        has_fps = args.fps
        width = args.width
        height = args.height
        has_display = args.display
        input_dir = args.input_dir
        output_dir = args.output_dir

        # Custom Params (refer to include/openpose/flags.hpp for more parameters)
        params = dict()
        params["model_folder"] = dir_path + "/../../../models/"
        params["hand"] = True
        params["hand_detector"] = 0
        params["body"] = 1

        # Add others in path?
        for i in range(0, len(args[1])):
            curr_item = args[1][i]
            if i != len(args[1])-1:
                next_item = args[1][i+1]
            else:
                next_item = "1"
            if "--" in curr_item and "--" in next_item:
                key = curr_item.replace('-', '')
                if key not in params:
                    params[key] = "1"
            elif "--" in curr_item and "--" not in next_item:
                key = curr_item.replace('-', '')
                if key not in params:
                    params[key] = next_item

        # Construct it from system arguments
        # op.init_argv(args[1])
        # oppython = op.OpenposePython()

        # check data
        data_dir = input_dir
        files = os.listdir(input_dir)
        print(type(files))
        # print(files)
        rs_frame_number = 1
        while True:
            color_frame_name = color_frame_name(rs_frame_number)
            depth_frame_name = depth_frame_name(rs_frame_number)
            if color_frame_name in files:
                # print(color_frame_name)
                if depth_frame_name not in files:
                    print(rs_frame_number)
            else:
                frames_total = rs_frame_number - 1
                break
            rs_frame_number += 1

        print(rs_frame_number)

        # Starting OpenPose
        opWrapper = op.WrapperPython()
        opWrapper.configure(params)
        opWrapper.start()

        # Processing Image
        datum = op.Datum()
        for i in range(frames_total):
            color_frame = color_frame_name(i)
            datum.cvInputData = color_frame
            opWrapper.emplaceAndPop(op.VectorDatum([datum]))

            # Display And Save Image
            print("Body keypoints: \n" + str(datum.poseKeypoints))
            cv2.imshow("OpenPose 1.7.0 - Tutorial Python API",
                       datum.cvOutputData)
            cv2.imwrite(str(i) + '.png', datum.cvOutputData)
            key = cv2.waitKey(0)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
        cv2.destroyAllWindows()
    except Exception as e:
        print(e)
        sys.exit(-1)
