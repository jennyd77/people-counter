import numpy as np
import cv2
import time

face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_alt.xml')
#person_cascade = cv2.CascadeClassifier('cascades/haarcascade_upperbody.xml')
#hog = cv2.HOGDescriptor()
#hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cap=cv2.VideoCapture(1)

while(True):
	# Capture frame by frame
	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
	for (x,y,w,h) in faces:
		print(x,y,w,h)
		cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)

	# Attempt to detect people using haarcascade_upperbody
	#people = person_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
	#	for (x,y,w,h) in people:
	#	cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
	# person_cascade was highly unsuccessful.

	# Attempt to detect people using HOG (Histograms of Oriented Gradients)
	#people, weights = hog.detectMultiScale(gray, winStride=(8,8) )
	#for (x,y,w,h) in people:
	#	print(x,y,w,h)
	#	cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
	# HOG (Histograms of Oriented Gradients) was quite unsuccessful too



	# Display the resulting frame
	cv2.imshow('frame', frame)
	if cv2.waitKey(20) & 0xFF == ord('q'):
		break
	#time.sleep(2)
# When complete, release the capture
cap.release()
cv2.destroyAllWindows()
