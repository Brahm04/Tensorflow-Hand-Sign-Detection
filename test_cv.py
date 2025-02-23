import cv2
import numpy as np

img = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.namedWindow("Test", cv2.WINDOW_NORMAL)  # Optional: allow window resizing
cv2.imshow("Test", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
