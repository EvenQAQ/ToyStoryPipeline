import pyrealsense2 as rs
import numpy as np
import cv2
import datetime
import time
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record video and picture")
    # add arguments
    parser.add_argument("-p", "--path", type=str,
                        default='../Data/', help="Path to save the data")
    parser.add_argument("-o", "--output", type=str,
                        help="Path to the output file (.bag)")
    args = parser.parse_args()
    save_path = args.path
    output_file = args.output

    start_time = datetime.datetime.now()
    num_frames = 0
    fps = 0
    num_image = 0
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
    # config.enable_record_to_file(save_path + output_file)
    # Start streaming
    pipeline.start(config)
    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            # Convert images to numpy arrays

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            # print(type(color_image))
            # print(color_image)

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(
                depth_image, alpha=0.03), cv2.COLORMAP_JET)
            # Stack both images horizontally
            images = np.hstack((color_image, depth_colormap))
            elapsed_time = (datetime.datetime.now() -
                            start_time).total_seconds()
            num_frames += 1
            fps = num_frames / elapsed_time
            cv2.putText(images, 'FPS: {}'.format(fps),
                        (40, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)
            key = cv2.waitKey(1)
            if key == 32:
                num_image += 1
                print('shoot' + str(num_image))
                cv2.imwrite('../Data/color' + output_file + str(num_image) +
                            '.png', color_image)
                cv2.imwrite('../Data/depth' + output_file + str(num_image) +
                            '.png', depth_image)
                cv2.imwrite('../Data/depth_colorMAP' +
                            output_file + str(num_image) + '.png', depth_colormap)

            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        # Stop streaming
        pipeline.stop()
