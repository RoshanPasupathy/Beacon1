# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 19:27:52 2015
@author: Roshan
"""

import numpy as np
import cv2
import os
import time 

os.system('v4l2-ctl -d 0 -c focus_auto=0')
os.system('v4l2-ctl -d 0 -c focus_absolute=0')
os.system('v4l2-ctl -d 0 -c zoom_absolute=100')
os.system('v4l2-ctl -d 0 -c exposure_auto=3')
#os.system('v4l2-ctl -d 0 -c exposure_absolute=120')
os.system('v4l2-ctl -d 0 -c contrast=128')
os.system('v4l2-ctl -d 0 -c brightness=128')
os.system('v4l2-ctl -d 0 -c white_balance_temperature_auto=1')
#os.system('v4l2-ctl -d 0 -c white_balance_temperature=6500')

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

def draw2(img,corners):
	img = cv2.rectangle(img,tuple(corners[0].ravel()),tuple(corners[20].ravel()),(0,255,0),2)
	return img

mtx = np.array([[ 620.54646,0, 316.17234],[0,621.10631,244.57960],[0,0,1]],dtype=np.float64)
dist = np.array([ 0.13359,-0.29557, -0.00070 ,-0.00054,0.00000 ],dtype=np.float64)
		
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 0.001)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
objp = 26.0 * objp
axis = np.float32([[78,0,0], [0,78,0], [0,0,-78.0]]).reshape(-1,3)
axis2 = np.float32([[50,20,90], [100,10,30], [45,25,-68.0]]).reshape(-1,3)

cap = cv2.VideoCapture(0)
cap.set(cv2.cv.CV_CAP_PROP_FPS, 30)
#a = cap.get(cv2.cv.CV_CAP_PROP_FPS)
b = cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
c = cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
print b,c
l = 0 
start = time.time()
while (True) & ( l < 1):
	ret,frame = cap.read()
	cv2.imshow('frame',frame)
	if cv2.waitKey(1) & 0xFF == ord('p'):
		print "stopped"
		frame1 = frame.copy()
		break
        if cv2.waitKey(1) & 0xFF == ord('q'):
		break
cap.release()
gray = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
if ret == True:
	print "ret true"
	cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
	#print corners
	rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners, mtx, dist)
	print rvecs
	print "Rmatrix"
	print cv2.Rodrigues(rvecs)
	print "translation vector"
	print tvecs
	imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
	imgpts2, jac2 = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
	print "imgpoints"
	print imgpts2
	#cv2.rectangle(frame1,tuple(corners[0].ravel()),tuple(corners[20].ravel()),(0,255,0),1)
	corner = tuple(corners[0].ravel())
	cv2.line(frame1, corner, tuple(imgpts[0].ravel()), (255,0,0), 2)
	cv2.line(frame1, corner, tuple(imgpts[1].ravel()), (0,255,0), 2)
	cv2.line(frame1, corner, tuple(imgpts[2].ravel()), (0,0,255), 2)
	cv2.imwrite('posecalibrate4.bmp',frame1)
	l = 1
	print 'image taken and saved'
else:
	"nope"
cv2.imshow('result',frame1)
end = time.time()
print end - start
print l/(end - start)
