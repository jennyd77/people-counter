import numpy as np
import cv2
import time
import gluoncv as gcv
from gluoncv.utils import try_import_cv2
cv2 = try_import_cv2()
import mxnet as mx

face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_alt.xml')
#person_cascade = cv2.CascadeClassifier('cascades/haarcascade_upperbody.xml')
#hog = cv2.HOGDescriptor()
#hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

net = gcv.model_zoo.get_model('ssd_512_mobilenet1.0_voc', pretrained=True)
net.hybridize()

cap=cv2.VideoCapture(0)

while(True):
	# Capture frame by frame
	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype('uint8')
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

	# Attempt to detect people using ssd_512_mobilenet1.0 and gluoncv
	frame_ndarray = mx.nd.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).astype('uint8')
	rgb_nd, frame_numpy = gcv.data.transforms.presets.ssd.transform_test(frame_ndarray, short=512, max_size=700)
	# Run frame through network
	class_IDs, scores, bounding_boxes = net(rgb_nd)
	#frame_numpy=frame_ndarray_tform.asnumpy()

	for i in range(len(scores[0])):
		#print(class_IDs.reshape(-1))
		#print(scores.reshape(-1))
		cid = int(class_IDs[0][i].asnumpy())
		cname = net.classes[cid]
		score = float(scores[0][i].asnumpy())
		if score < 0.5:
			break
		x,y,w,h = bbox =  bounding_boxes[0][i].astype(int).asnumpy()
		print(cname, score, bbox)
		tag = "{}: {:.4f}".format(cname, score)
		cv2.rectangle(frame_numpy, (x,y), (w, h), (0, 255, 0), 2)
		cv2.putText(frame_numpy, tag, (x, y-20),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1)

	# Display the resulting frame
	cv2.imshow('frame', frame_numpy)
	if cv2.waitKey(20) & 0xFF == ord('q'):
		break
	#time.sleep(2)
# When complete, release the capture
cap.release()
cv2.destroyAllWindows()
