#!/usr/bin/env python
import cv2

# Try to open each camera index and grab a frame
for index in range(0, 40):  # Adjust the range depending on how many cameras you expect
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            cv2.imshow(f"Camera Index: {index}", frame)
            cv2.waitKey(0)
            cap.release()
        else:
            print(f"No frame available for camera at index {index}")
    else:
        print(f"Cannot open camera at index {index}")

cv2.destroyAllWindows()
