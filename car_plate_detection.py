#basic_setup
#from envs.cv import cv2
#from "C:\Users\Cathy\anaconda3\envs\cv" import cv2
#import cv2
#import numpy as np
#example: https://www.kaggle.com/datasets/smeschke/four-shapes?select=process_data.py

import cv2
import numpy as np
import imutils
from matplotlib import pyplot as plt
import pytesseract

img = cv2.imread('LPdata/images/Cars10.png')
img = cv2.resize(img, (620, 480))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray, 13,15,15)
edged = cv2.Canny(gray, 30, 200) #Edge detection

contours = cv2.findContours(edged.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
contours = sorted(contours,key=cv2.contourArea, reverse = True)[:10]
screenCnt = None
for c in contours:
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.018 * peri, True)
    # if our approximated contour has four points, then
    # we can assume that we have found our screen
    if len(approx) == 4:
        screenCnt = approx
        break
print(screenCnt)
img2 = cv2.polylines(img,[screenCnt],True,(255,0,0))
mask = np.zeros(gray.shape,np.uint8)


# Masking the part other than the number plate
mask = np.zeros(gray.shape,np.uint8)
new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
#new_image = cv2.bitwise_and(img,img,mask=mask)

plt.imshow(new_image)
plt.show()