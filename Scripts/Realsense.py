#
# 作者：jacob-srs
# 文件：学习 realsense 3
# 模块：
# 版本：v1.0
# 功能：保存图像
#


import cv2
import numpy as np

import pyrealsense2 as rs


class Realsense():
    """ 调用 realsense sdk 开发的相机代码 """
    # def __init__(self, width=640, height=480, fps=30):

    def __init__(self, resolution=(640, 480), fps=60):
        """ 可自定义相机分辨率 (col x row)

        intelRealsense fact sheet:
        - Depth Stream Output Resolution: Up to 1280 x 720
        - Depth Stream Output Frame Rate: Up to 90 fps
        - Minimum Depth Distance (Min-Z): 0.16 m
        - Maximum Range: Approx. 10 meters
        - RGB Sensor Resolution & Frame Rate: 1920 x 1080 at 30 fps
        """
        col = resolution[0]
        row = resolution[1]
        fps = fps

        self.resolution = (col, row)
        self.config = None
        self.pipeline = None
        self.rs_start_flag = False
        self.align = None

        self._get_realsense_device(self.resolution)
        self._configure_stream(col, row, fps)
        self._start_pipeline()

    def _get_realsense_device(self, resolution):
        """ 查找连接的 realsense 设备数目 """

        import os
        import sys

        # Get a snapshot of currently connected devices
        ctx = rs.context()
        device_list = ctx.query_devices()
        # # if len(device_list) == 0:
        # print(device_list)  # <pyrealsense2.pyrealsense2.device_list object at 0x0000014C6F4F3298>
        # print(len(device_list))  # 1
        # for dev in device_list:
        #     print(dev)  # <pyrealsense2.device: Intel RealSense D415 (S/N: 915422060024)>
        # print(device_list.size())  # 1，size() numpy 函数
        # print(type(device_list))  # <class 'pyrealsense2.pyrealsense2.device_list'>
        if device_list.size == 0:
            print("no realsense devices found... ")
            os.system("pause")
            sys.exit(0)
        else:  # 打印设备信息
            for i, device in enumerate(device_list):
                i += 1
                device_name = device.get_info(rs.camera_info.name)
                print("found [{0}] device(s) required :: [{1}]".format(
                    i, device_name))
                print("resolution :: {0}".format(resolution))

    def _configure_stream(self, col, row, fps):
        """ 配置深度和彩色数据流 """
        self.config = rs.config()
        self.config.enable_stream(
            rs.stream.depth, col, row, rs.format.z16, fps)
        self.config.enable_stream(
            rs.stream.color, col, row, rs.format.bgr8, fps)

    def _start_pipeline(self):
        """ 创建 realsense 管道，简化使用者与设备的交互 """
        self.pipeline = rs.pipeline()
        self.pipeline.start(self.config)
        self.rs_start_flag = True

        # 创建一个 align 对象，将深度帧与其他帧对齐
        align_to = rs.stream.color  # 与彩色帧对齐
        self.align = rs.align(align_to)

    def _rs_read_frames(self):
        """ 读取 realsense 的视频帧，返回帧:颜色 + 深度 """
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        # depth_frame, color_frame 格式?
        return (depth_frame, color_frame)

    def _rs_read_frames_align(self):
        """ 读取 realsense 对齐后的视频帧，返回帧：颜色 + RGB """
        frames = self.pipeline.wait_for_frames()

        # align the depth frame to color frame
        aligned_frames = self.align.process(frames)

        # get color frame + aligned depth frame
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        return (aligned_depth_frame, color_frame)

    # def rs_get_image(self, enable_align=True):
    def rs_get_image(self, enable_align=False):
        """ 得到 realsense 的图像，返回图像：颜色 + 深度 + 聚合图 """
        if enable_align is True:
            depth_frame, color_frame = self._rs_read_frames_align()
        else:
            depth_frame, color_frame = self._rs_read_frames()

        # 得到 depth_data, color_data
        depth_data = depth_frame.get_data()
        color_data = color_frame.get_data()

        # 转换成数组并得到图像
        depth_image = np.asanyarray(depth_data)
        color_image = np.asanyarray(color_data)

        # 得到深度图的伪彩色图
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03),
            cv2.COLORMAP_JET)

        # 聚合图像
        stack_image = np.hstack((color_image, depth_colormap))

        # cv2.imshow("color", color_image)
        # key = cv2.waitKey(1)
        # if key == 27:
        #     cv2.destroyAllWindows()

        return (depth_image, color_image, stack_image)
        # return depth_colormap, color_image

    def close(self):
        """ 关闭 realsense """
        if self.rs_start_flag is False:
            print("realsense camera already turned off")
        else:  # flag is True
            self.pipeline.stop()
            self.rs_start_flag = False

    def save(self):
        """ 保存功能 """
        pass


if __name__ == "__main__":
    def turn_on_camera(device):
        import cv2

        while True:
            depth_image, color_image, stack_image = device.rs_get_image()

            cv2.namedWindow("Realsense", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("Realsense", stack_image)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                # 是用 break 跳出后 进入 close，还是直接调用 close？
                # device.close()  # 出现 wait_for_frame before start
                break

    image_size = (640, 480)
    realsense_camera = Realsense(image_size)
    turn_on_camera(realsense_camera)
    realsense_camera.close()
