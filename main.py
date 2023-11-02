import os
import cv2 as cv
import numpy as np
from Stitcher import Stitcher

IMAGES_PARENT_DIR = "slike/"
RATIO = 0.75
THRESHOLD = 0.995

stitcher = Stitcher()
images = []
j = 0
for index in range(1, len(os.listdir(IMAGES_PARENT_DIR))):
    path = IMAGES_PARENT_DIR + "Picture" + str(index) + ".jpg"
    image = cv.imread(path)
    images.append(image)
    if j == 20:
        break
    j += 1

result = None
index = 0

while index < len(images):
    if index == 0:
        result = stitcher.stitch([images[index], images[index + 1]], RATIO, THRESHOLD)
        index += 2
        continue
    result = stitcher.stitch([result, images[index]], RATIO, THRESHOLD)
    index += 1

try:
    cv.imwrite("result_image_new.jpg", result)
    cv.waitKey(0)
except Exception as e:
    print(f"Error: {format(e)}")

cv.destroyAllWindows()
