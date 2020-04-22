import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')

cap=cv2.VideoCapture(0)

while(True):
	# Capture frame by frame
	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(frame, scaleFactor=1.5, minNeighbors=5)
	for (x,y,w,h) in faces:
		print(x,y,w,h)
	# Display the resulting frame
	cv2.imshow('frame', frame)
	if cv2.waitKey(20) & 0xFF == ord('q'):
		break

# When complete, release the capture
cap.release()
cv2.destroyAllWindows()
