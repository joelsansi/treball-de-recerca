# USAGE
# python recognize_video.py -d face_detection_model -m openface_nn4.small2.v1.t7 -r output/recognizer.pickle -l output/le.pickle

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
from datetime import datetime
import pandas as pd
import numpy as np
import pygsheets
import imutils
import pickle
import socket
import time
import cv2
import os

#Google Sheets stuf
gc = pygsheets.authorize(service_file='sheets.json')
sheet = gc.open("Confirmed Detections")
worksheet = sheet[0]

#----------------------------------- TCP SETUP -------------------------------------
host = "192.168.0.29" #socket.gethostbyname(socket.gethostname()) 
port = 50000


s = socket.socket()
#-------------------FOR TCP SERVER ---------------------
#print("host: " + host)
#s.bind((host,port))
#s.listen(1)
#client_socket, address = s.accept()
#print("Connection from: " + str(address))

#--------------------FOR TCP CLIENT---------------------
s.connect((host,port))


#global variables for information stock
times_detected = {}
confirmed_detections = {"Nom":[],"Data": [], "Hora":[]}
unknow_count = 0
unknow_time = 0
timedate = ""
current_date = ""
current_time = ""

#Send data variables
send_loop_count = 0
send_state = False
data = ""

args = {'detector': 'face_detection_model',
'embedding_model': 'openface_nn4.small2.v1.t7',
'recognizer': 'output/recognizer.pickle',
'le': 'output/le.pickle',
'confidence': 0.5}

# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load our serialized face embedding model from disk
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open(args["recognizer"], "rb").read())
le = pickle.loads(open(args["le"], "rb").read())

# initialize the video stream, then allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# start the FPS throughput estimator
fps = FPS().start()

# loop over frames from the video file stream

while True:
	# grab the frame from the threaded video stream
	frame = vs.read()

	# resize the frame to have a width of 720 pixels (while
	# maintaining the aspect ratio), and then grab the image
	# dimensions
	frame = imutils.resize(frame, width=720)
	(h, w) = frame.shape[:2]

	# construct a blob from the image
	imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(frame, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)

	# apply OpenCV's deep learning-based face detector to localize
	# faces in the input image
	detector.setInput(imageBlob)
	detections = detector.forward()

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections
		if confidence > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for
			# the face
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# extract the face ROI
			face = frame[startY:endY, startX:endX]
			(fH, fW) = face.shape[:2]

			# ensure the face width and height are sufficiently large
			if fW < 20 or fH < 20:
				continue

			# construct a blob for the face ROI, then pass the blob
			# through our face embedding model to obtain the 128-d
			# quantification of the face
			faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
				(96, 96), (0, 0, 0), swapRB=True, crop=False)
			embedder.setInput(faceBlob)
			vec = embedder.forward()

			# perform classification to recognize the face
			preds = recognizer.predict_proba(vec)[0]
			j = np.argmax(preds)
			proba = preds[j]
			name = le.classes_[j]

			# draw the bounding box of the face along with the
			# associated probability
			text = "{}: {:.2f}%".format(name, proba * 100)
			y = startY - 10 if startY - 10 > 10 else startY + 10
			cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)

			cv2.putText(frame, text, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

			if name != "Desconegut":
				if name not in times_detected:
					times_detected[name] = [1,time.perf_counter()]

				elif name in times_detected:
					times_detected[name][0] += 1
				
				if time.perf_counter() - times_detected[name][1] > 5:
					times_detected[name][1] = time.perf_counter()
					times_detected[name][0] = 1
				
				elif times_detected[name][0] >=  30:
					times_detected[name][0] = 0
					times_detected[name][1] = 0
					print(f"{name} ha sigut reconegut/da")
					data = name
					send_state = True
					

					
					if name not in confirmed_detections:
						date_time = datetime.now()
						confirmed_detections["Nom"].append(name)
						confirmed_detections["Hora"].append(date_time.strftime("%H:%M"))
						confirmed_detections["Data"].append(date_time.strftime("%d/%m/%Y"))
						df = pd.DataFrame(confirmed_detections)
						worksheet.set_dataframe(df,(1,1))


			elif name == "Desconegut":
				if unknow_count == 0:
					unknow_time = time.perf_counter()
					unknow_count += 1
				else:
					unknow_count += 1

				if time.perf_counter() - unknow_time > 5:
					unknow_time = time.perf_counter
					unknow_count = 0

				
				if unknow_count > 30:
					unknow_time = time.perf_counter
					unknow_count = 0
					print("Ho sento, no puc reconèixer-te")
					data = "Ho sento no he pogut reconèixer-te"
					send_state = True


			if send_state:
				send_loop_count += 1
				s.send(str(data).encode())
				if send_loop_count == 50:
					send_loop_count = 0
					data = ""
					send_state = False

			else:
				s.send(str(data).encode())


	# update the FPS counter
	fps.update()

	# show the output frame
	cv2.imshow("Reconeixedor facial", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		s.send("exit".encode())
		s.close()
		break

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

print("Persones detectadas: ")
for  names in confirmed_detections["Nom"]:
	print(names)