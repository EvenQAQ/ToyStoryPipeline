# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse

script_dir = 'D://DevTools/openpose/build/examples/tutorial_api_python'

try:
    # Import Openpose (Windows/Ubuntu/OSX)
    dir_path = script_dir
    print(dir_path)
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
            sys.path.append('../../python')
            # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
            # sys.path.append('/usr/local/python')
            from openpose import pyopenpose as op
    except ImportError as e:
        print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e

    # Flags
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
    video_source = args[0].video_source
    has_fps = args[0].fps
    width = args[0].width
    height = args[0].height
    has_display = args[0].display
    input_dir = args[0].input_dir
    output_dir = args[0].output_dir
    openpose_dir = output_dir + '/OpenPoseData'
    mediapipe_dir = output_dir + '/MediaPipeData'
    f = open(output_dir + '/print.txt', 'a')

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

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()
    cap = cv2.VideoCapture(video_source)
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue
        # Process Image
        datum = op.Datum()
        # imageToProcess = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        imageToProcess = cv2.flip(image, 1)
        datum.cvInputData = imageToProcess
        opWrapper.emplaceAndPop([datum])

        # Display Image
        f.writelines("Body keypoints: \n" + str(datum.poseKeypoints) + '\n')
        cv2.imshow("OpenPose 1.6.0 - Tutorial Python API", datum.cvOutputData)
        key = cv2.waitKey(15)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
except Exception as e:
    print(e)
    sys.exit(-1)
