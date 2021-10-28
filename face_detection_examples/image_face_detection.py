
# import the necessary packages
import numpy as np
import cv2

args = {'image': 'stock-image.jpg', # change that to use a different image (according to filename.extension)
'prototxt': 'deploy.prototxt.txt', 
'model': 'res10_300x300_ssd_iter_140000.caffemodel', 
'confidence': 0.5}

# load model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# load image and store its size
image = cv2.imread(args["image"])
(h, w) = image.shape[:2]

# transform the image in standard size if is too big
if w > 720:
	aspect_ratio = w/720
	final_w = round(w/aspect_ratio)
	final_h = round(h/aspect_ratio)
	output = cv2.resize(image,(final_w,final_h))
	h, w = final_h, final_w

else:
	output = image

# build blob with an 300x300 resized image
blob = cv2.dnn.blobFromImage(cv2.resize(output, (300, 300)), 1.0,
	(300, 300), (104.0, 177.0, 123.0))

# pass the blob through the network and obtain the detections and predictions
print("[INFO] computing object detections...")
net.setInput(blob)
detections = net.forward()

# loop over the detections
for i in range(0, detections.shape[2]):
	# extract the confidence (i.e., probability) associated with the
	# prediction
	confidence = detections[0, 0, i, 2]

	# filter out weak detections by ensuring the `confidence` is
	# greater than the minimum confidence
	if confidence > args["confidence"]:
		# compute the (x, y)-coordinates of the bounding box for the
		# object
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")
 
		# draw the bounding box of the face along with the associated
		# probability
		text = "{:.2f}%".format(confidence * 100)
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.rectangle(output, (startX, startY), (endX, endY),
			(0, 0, 255), 2)
		cv2.putText(output, text, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

# save the output
cv2.imwrite("output.png",output)			

# show the output image
cv2.imshow("Output" ,output)
cv2.waitKey(0)

