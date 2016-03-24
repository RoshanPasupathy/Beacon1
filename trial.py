import numpy as np
import cv2
import os
from LUTbeacontest1 import squarelut8
from LUTbeacontest1 import cleanupf
from LUTbeacontest1 import final_return 
import time
from threading import Thread
import socket

os.system('v4l2-ctl -d 0 -c focus_auto=0')
os.system('v4l2-ctl -d 0 -c focus_absolute=0')
os.system('v4l2-ctl -d 0 -c exposure_auto=1')
os.system('v4l2-ctl -d 0 -c exposure_absolute=3')
os.system('v4l2-ctl -d 0 -c contrast=100')
os.system('v4l2-ctl -d 0 -c brightness=100')
os.system('v4l2-ctl -d 0 -c white_balance_temperature_auto=0')
os.system('v4l2-ctl -d 0 -c white_balance_temperature=6500')
#cap = cv2.VideoCapture(0)
#cap.set(cv2.cv.CV_CAP_PROP_FPS, 30)
#a = cap.get(cv2.cv.CV_CAP_PROP_FPS)
#b = cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
#c = cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
#print b,c
output = np.array([0,640,0,480,0,480])
##########################
mtx = np.array([[ 619.66761,0, 316.18988],[0,620.24444,244.36804],[0,0,1]],dtype=np.float64)
distcoeff = np.array([0.13305,-0.29345,-0.00073,-0.00054,0.00000],dtype=np.float64)
R = np.array([[  1.17775762e-02,   9.99924734e-01,  -3.43731733e-03],
       [ -9.41574859e-01,   1.22473273e-02,   3.36581027e-01],
       [  3.36597792e-01,  -7.27617111e-04,   9.41648234e-01]])
tvec = np.array([[-210.1362324],[225.03386539],[1085.97995573]])
#########################
Q = np.dot(mtx,R)
#calculate A  * translation. last column of matrix
q = np.dot(mtx,tvec)
#calculate inverse of Q
Qinv = np.linalg.inv(Q)
#calculate inverse of Q 
_Qinv = -1.0*Qinv
#calculate c vector 3 x 1
c = np.dot(_Qinv,q)
failarr = np.array([[0],[0],[0]],dtype=np.float64)

class WebcamVideoStream:
	def __init__(self, src=0):
		# initialize the video camera stream and read the first frame
		# from the stream
		self.stream = cv2.VideoCapture(src)
		(self.grabbed, self.frame) = self.stream.read()

		# initialize the variable used to indicate if the thread should
		# be stopped
		self.stopped = False

	def start(self):
		# start the thread to read frames from the video stream
		t = Thread(target=self.update, args=())
		t.daemon = True
		t.start()
		return self

	def update(self):
		# keep looping infinitely until the thread is stopped
		while True:
			# if the thread indicator variable is set, stop the thread
			if self.stopped:
				return

			# otherwise, read the next frame from the stream
			(self.grabbed, self.frame) = self.stream.read()

	def read(self):
		# return the frame most recently read
		return self.frame

	def stop(self):
		# indicate that the thread should be stopped
		self.stopped = True
#soc = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
#soc.connect(('192.168.42.1',8000))

#soc.send('c' + c.dumps()) #len156
print "c is",c
vs = WebcamVideoStream(src=0).start()
l = 1
i = 1
start = time.clock()
while (True) & ( l < 400):
	#ret,frame = cap.read()
	frame = vs.read()
	output = squarelut8(output,480,640,10,frame[output[4]:output[5],:,:])
	if output[0] <= output[1]:
		cv2.rectangle(frame,(output[0],output[2]),(output[1],output[3]),(255,0,0),2)
		print "detected points",np.asarray(output)[0:4]
		u = final_return(np.asarray(output)[0:4], Qinv)
		print "u is",u
		#soc.send('u' + u.dumps())
	else:
		#soc.send('d' + failarr.dumps())
		print "Ball Not detected"
	l += 1
	cv2.imshow('frame',frame)
	#if cv2.waitKey(1) & 0xFF == ord('c'):
	#	stringval = 'img' + str(i) +'.bmp'
	#	cv2.imwrite(stringval,frame1)
	#	i += 1
	#	print stringval + ' taken'
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
end = time.clock()
print 'time taken =',end - start,'seconds'
print 'frame rate =',l/(end-start)
#soc.send('e' + failarr.dumps())

#soc.close()
cleanupf()
#cap.release()
cv2.destroyAllWindows()
vs.stop()
