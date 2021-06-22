import cv2
from cv2 import dnn_superres


sr = dnn_superres.DnnSuperResImpl_create()
image = cv2.imread('./tes.jpg')

sr.readModel("EDSR_x3.pb")
sr.setModel("edsr", 3)
result = sr.upsample(image)

cv2.imshow('result',result)
cv2.imshow('init', image)
cv2.waitKey()

#pip install opencv-contrib-python
#https://towardsdatascience.com/deep-learning-based-super-resolution-with-opencv-4fd736678066