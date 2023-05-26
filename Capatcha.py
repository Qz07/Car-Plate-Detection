import cv2
import numpy as np
import imutils
from matplotlib import pyplot as plt
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Cathy\anaconda3\envs\cv\Lib\site-packages\pytesseract\pytesseract.exe'


img = cv2.imread('2b827.png')

dst = cv2.fastNlMeansDenoisingColored(img,None,20,20,7,21)
dst = cv2.bilateralFilter(dst, 13, 20, 20)

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# mask = np.zeros(gray.shape,np.uint8)
# (x, y) = np.where(mask == 255)
# (topx, topy) = (np.min(x), np.min(y))
# (bottomx, bottomy) = (np.max(x), np.max(y))
# Cropped = gray[topx:bottomx+1, topy:bottomy+1]
plt.subplot(121),plt.imshow(img)
plt.subplot(122),plt.imshow(dst)
# text = pytesseract.image_to_string(dst)
# print("Number is:",text)
# plt.imshow(Cropped)
plt.show()