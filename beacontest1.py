import numpy as np
import cv2
import os
from LUTbeacontest1 import squarelut8
from LUTbeacontest1 import cleanupf
from LUTbeacontest1 import final_return 
import time
from threading import Thread
import socket
import sys,traceback

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

mtx = np.array([[ 620.54646,0, 316.17234],[0,621.10631,244.57960],[0,0,1]],dtype=np.float64)
distcoeff = np.array([ 0.13359,-0.29557, -0.00070 ,-0.00054,0.00000 ],dtype=np.float64)
R = np.array([[  5.08578383e-04,   9.99432156e-01,  -3.36913444e-02],
       [ -9.32515472e-01,   1.26409351e-02,   3.60908716e-01],
       [  3.61129666e-01,   3.12341496e-02,   9.31992378e-01]])
tvec = np.array([[-204.29808184],[187.40871596],[648.37443412]])


#########################
Q = np.dot(mtx,R)
#calculate A  * translation. last column of matrix
q = np.dot(mtx,tvec)
#calculate inverse of Q
Qinv = np.linalg.inv(Q)
#calculate inverse of Q 
_Qinv = -1.0*Qinv
#calculate c vector 3 x 1
c = np.dot(_Qinv,q).ravel()
failarr = np.array([0,0,0],dtype=np.float64)

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

try:
	soc = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
	soc.connect(('192.168.42.1',8000))
	
	soc.send(''.join(['c',c.tostring(),'l'])) #len156
	
	vs = WebcamVideoStream(src=0).start()
	failtimes = 0
	l = 1
	i = 1
	start = time.clock()
	while (True) & ( l < 3000):
		#ret,frame = cap.read()
		frame = vs.read()
		output = squarelut8(output,480,640,10,frame[output[4]:output[5],:,:])
		if output[0] <= output[1]:
			#cv2.rectangle(frame,(output[0],output[2]),(output[1],output[3]),(255,0,0),2)
			print np.asarray(output)[0:4]
			u = final_return(np.asarray(output)[0:4], Qinv).ravel()
			soc.send(''.join(['u',u.tostring(),'l']))
			failtimes = 0
		else:
			failtimes += 1
			if failtimes > 5:
				soc.send(''.join(['d',failarr.tostring(),'l']))
				failtimes = 0
			print "Ball Not detected"
		l += 1
		#time.sleep(0.001)
		#cv2.imshow('frame',frame)
		#if cv2.waitKey(1) & 0xFF == ord('c'):
		#	stringval = 'img' + str(i) +'.bmp'
		#	cv2.imwrite(stringval,frame1)
		#	i += 1
		#	print stringval + ' taken'
		#if cv2.waitKey(1) & 0xFF == ord('q'):
		#	break
	end = time.clock()
	print 'time taken =',end - start,'seconds'
	print 'frame rate =',l/(end-start)
	soc.send(''.join(['e',failarr.tostring(),'l']))

except:
	typ,val,tb = sys.exc_info()
	traceback.print_exception(typ,val,tb)

	
finally:	
	soc.close()
	cleanupf()
	#cap.release()
	cv2.destroyAllWindows()
	vs.stop()
	
