import argparse
import glob
import os

import cv2

from yolo import YOLO

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--images', default="images",
                help='Path to images or image file')
ap.add_argument('-n', '--network', default="normal",
                help='Network Type: normal / tiny / prn / v4-tiny')
ap.add_argument('-d', '--device', default=0, help='Device to use')
ap.add_argument('-s', '--size', default=416, help='Size for yolo')
ap.add_argument('-c', '--confidence', default=0.25, help='Confidence for yolo')
args = ap.parse_args()

if args.network == "normal":
    print("loading yolo...")
    yolo = YOLO("models/cross-hands.cfg",
                "models/cross-hands.weights", ["hand"])
elif args.network == "prn":
    print("loading yolo-tiny-prn...")
    yolo = YOLO("models/cross-hands-tiny-prn.cfg",
                "models/cross-hands-tiny-prn.weights", ["hand"])
elif args.network == "v4-tiny":
    print("loading yolov4-tiny-prn...")
    yolo = YOLO("models/cross-hands-yolov4-tiny.cfg",
                "models/cross-hands-yolov4-tiny.weights", ["hand"])
else:
    print("loading yolo-tiny...")
    yolo = YOLO("models/cross-hands-tiny.cfg",
                "models/cross-hands-tiny.weights", ["hand"])

yolo.size = int(args.size)
yolo.confidence = float(args.confidence)

print("extracting tags for each image...")
if args.images.endswith(".txt"):
    with open(args.images, "r") as myfile:
        lines = myfile.readlines()
        files = map(lambda x: os.path.join(
            os.path.dirname(args.images), x.strip()), lines)
else:
    files = sorted(glob.glob("%s/*.jpg" % args.images))

conf_sum = 0
detection_count = 0
file_list = ['/Users/evenqaq/Dev/Codes/ToyStory/Data/Data/color11.png',
             '/Users/evenqaq/Dev/Codes/ToyStory/Data/Data/color118.png', '/Users/evenqaq/Pictures/证件照.jpg']
for idx, file in enumerate(file_list):
    print(file)
    mat = cv2.imread(file)

    width, height, inference_time, results = yolo.inference(mat)

    print("%s in %s seconds: %s classes found!" %
          (os.path.basename(file), round(inference_time, 2), len(results)))

    output = []

    cv2.namedWindow(str(file), cv2.WINDOW_NORMAL)
    cv2.resizeWindow(str(file), 640, 480)

    for detection in results:
        id, name, confidence, x, y, w, h = detection
        final_confidence = round(confidence, 2)
        if final_confidence < 0.9:
            continue
        else:
            cx = x + (w / 2)
            cy = y + (h / 2)

            conf_sum += confidence
            detection_count += 1

            # draw a bounding box rectangle and label on the image
            color = (255, 0, 255)
            cv2.rectangle(mat, (x, y), (x + w, y + h), color, 1)
            text = "%s (%s)" % (name, round(confidence, 2))
            cv2.putText(mat, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.25, color, 1)

            print("%s with %s confidence" % (name, round(confidence, 2)))

            cv2.imwrite(str(idx) + "export.png", mat)

    # show the output image
    cv2.imshow(str(file), mat)
    key = cv2.waitKey(0)
    if key & 0xFF == ord('q'):
        continue
    if key == 27:  # exit on ESC
        break


print("AVG Confidence: %s Count: %s" %
      (round(conf_sum / detection_count, 2), detection_count))
cv2.destroyAllWindows()
