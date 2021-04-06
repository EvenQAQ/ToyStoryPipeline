import cv2
import os
import sys
import argparse
import numpy as np


def is_MP4_file(file):
    print(file)
    suffix = os.path.splitext(file)[1]
    if suffix == '.mp4':
        return True

    return False


def get_cap_serial(file):
    suffix = os.path.splitext(file)[0]
    print(type(suffix))
    if '809512060382' in suffix:
        return 0
    if '825312071677' in suffix:
        return 1
    if '834412071317' in suffix:
        return 2


locs = ['head', 'body', 'leftarm', 'rightarm', 'leftleg', 'rightleg']
dirs = {}
files = []
cams = [cv2.VideoCapture() for i in range(3)]
frames = []
cam_num = 0
num_image = 0
num_head = 0
num_body = 0
num_leftarm = 0
num_rightarm = 0
num_leftleg = 0
num_rightleg = 0
num_cap = 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str,
                        default="", help="input dir")
    args = parser.parse_args()
    input_dir = args.input + '/'
    for i in locs:
        dirs[i] = input_dir + i + '/'
        if not os.path.exists(dirs[i]):
            os.mkdir(dirs[i])
    print(input_dir)
    for i in os.listdir(input_dir):
        if is_MP4_file(i):
            cams[get_cap_serial(i)] = cv2.VideoCapture(input_dir + i)

    for i in cams:
        i.set(cv2.CAP_PROP_FPS, 60)

    while cams[0].isOpened():
        ret0, frame0 = cams[0].read()
        ret1, frame1 = cams[1].read()
        ret2, frame2 = cams[2].read()
        num_cap += 1
        frame = np.hstack((frame0, frame1, frame2))

        cam_num = 0
        cv2.namedWindow('preview')

        cv2.imshow('preview', frame)
        # print(cams[0].get(1))
        key = cv2.waitKey(20)
        if key & 0xFF == ord('u'):
            cv2.imwrite(dirs['head'] + str(num_head) + '.png', frame)
            num_image += 1
            num_head += 1
            print('head' + str(num_head) + '/' + str(num_image))
            # break
        elif key & 0xFF == ord('j'):
            cv2.imwrite(dirs['body'] + str(num_body) + '.png', frame)
            num_image += 1
            num_body += 1
            print('body' + str(num_body) + '/' + str(num_image))
        elif key & 0xFF == ord('h'):
            cv2.imwrite(dirs['leftarm'] + str(num_leftarm) + '.png', frame)
            num_image += 1
            num_leftarm += 1
            print('leftarm' + str(num_leftarm) + '/' + str(num_image))
        elif key & 0xFF == ord('k'):
            cv2.imwrite(dirs['rightarm'] + str(num_rightarm) + '.png', frame)
            num_image += 1
            num_rightarm += 1
            print('rightarm' + str(num_rightarm) + '/' + str(num_image))
        elif key & 0xFF == ord('n'):
            cv2.imwrite(dirs['leftleg'] + str(num_leftleg) + '.png', frame)
            num_image += 1
            num_leftleg += 1
            print('leftleg' + str(num_leftleg) + '/' + str(num_image))
        elif key & 0xFF == ord('m'):
            cv2.imwrite(dirs['rightleg'] + str(num_rightleg) + '.png', frame)
            num_image += 1
            num_rightleg += 1
            print('rightleg' + str(num_rightleg) + '/' + str(num_image))

        if key == 32:
            while True:
                key = cv2.waitKey(0)
                if key & 0xFF == ord('u'):
                    cv2.imwrite(dirs['head'] + str(num_head) + '.png', frame)
                    num_image += 1
                    num_head += 1
                    print('head' + str(num_head) + '/' + str(num_image))
                    # break
                elif key & 0xFF == ord('j'):
                    cv2.imwrite(dirs['body'] + str(num_body) + '.png', frame)
                    num_image += 1
                    num_body += 1
                    print('body' + str(num_body) + '/' + str(num_image))
                elif key & 0xFF == ord('h'):
                    cv2.imwrite(dirs['leftarm'] +
                                str(num_leftarm) + '.png', frame)
                    num_image += 1
                    num_leftarm += 1
                    print('leftarm' + str(num_leftarm) + '/' + str(num_image))
                elif key & 0xFF == ord('k'):
                    cv2.imwrite(dirs['rightarm'] +
                                str(num_rightarm) + '.png', frame)
                    num_image += 1
                    num_rightarm += 1
                    print('rightarm' + str(num_rightarm) + '/' + str(num_image))
                elif key & 0xFF == ord('n'):
                    cv2.imwrite(dirs['leftleg'] +
                                str(num_leftleg) + '.png', frame)
                    num_image += 1
                    num_leftleg += 1
                    print('leftleg' + str(num_leftleg) + '/' + str(num_image))
                elif key & 0xFF == ord('m'):
                    cv2.imwrite(dirs['rightleg'] +
                                str(num_rightleg) + '.png', frame)
                    num_image += 1
                    num_rightleg += 1
                    print('rightleg' + str(num_rightleg) + '/' + str(num_image))
                elif key & 0xFF == ord('a'):
                    num_cap -= 1
                    print('press left')
                elif key & 0xFF == ord('d'):
                    num_cap += 1
                    print('press right')
                elif key == 32:
                    print('press space')
                    break
                for i in cams:
                    i.set(1, num_cap)
                ret0, frame0 = cams[0].read()
                ret1, frame1 = cams[1].read()
                ret2, frame2 = cams[2].read()

                frame = np.hstack((frame0, frame1, frame2))
                cv2.imshow('preview', frame)
                if key & 0xFF == ord('q') or key == 27:
                    cv2.destroyAllWindows()
                    break

        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
    for i in cams:
        i.release()
    cv2.destroyWindow("preview")
