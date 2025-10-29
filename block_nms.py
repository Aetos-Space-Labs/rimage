import numpy as np
import rimage
import time
import cv2

IMAGE_PATH = "image.png"
TOTAL_POINTS = 10
DIST_BLOCKS = 2
SQUARE = 20

img_gray = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
h, w = img_gray.shape

nms = rimage.PyBlockNms(SQUARE, h, w)
stamp = time.time()
points = nms.run(img_gray.astype(np.uint16), DIST_BLOCKS, TOTAL_POINTS)
print(time.time() - stamp)

img_vis = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
for (x, y) in points: cv2.circle(img_vis, (x, y), 3, (0, 0, 255), -1)

cv2.imshow("Detects", img_vis)
cv2.waitKey(0)
cv2.destroyAllWindows()