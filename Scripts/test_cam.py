import cv2




cv2.namedWindow("preview")
vc = cv2.VideoCapture('../Data/ysc10131403/ysc-825312071677.mp4')
vc.set(cv2.CAP_PROP_FPS, 60)
print (vc.get(5))
if vc.isOpened():  # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    cv2.imshow("preview", frame)
    rval, frame = vc.read()
    key = cv2.waitKey(0)
    print(key)
vc.release()
cv2.destroyWindow("preview")
