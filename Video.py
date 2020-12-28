from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2

# buat argumen parse dan parse argumen
ap = argparse.ArgumentParser()
ap.add_argument("-p","--prototxt", required=True,
    help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m","--model",required=True,
    help="path to Caffe pre-trained model")
ap.add_argument("-c","--confidence",type=float,default=0.2,
    help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# menginisialisasi daftar label kelas yang dilatih oleh MobileNet SSD
# detect, lalu buat satu set warna kotak pembatas untuk setiap kelas
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "manusia", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0,255,size=(len(CLASSES),3))

# memuat model serial dari disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"],args["model"])

# menginisialisasi jalur video, cammera melakukan pemanasan
# dan menginisialisasi penghitung FPS
print("[INFO] starting vidio stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

# loop di atas bingkai dari jalur video
while True:
    # ambil bingkai dari jalur video berulir dan ubah ukurannya
    # memiliki lebar maksimum 400 piksel

    frame = vs.read()
    frame = imutils.resize(frame, width=1080)

    # ambil dimensi bingkai dan ubah menjadi blob
    (h,w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)),
        0.007843, (300,300), 127.5)

    # lewati blob melalui jaringan dan dapatkan deteksi dan prediksi
    net.setInput(blob)
    detections = net.forward()

    # loop di atas terdeteksi
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0,0,i,2]
        if confidence > args["confidence"]:
            idx = int(detections[0,0,i,1])
            box = detections[0,0,i,3:7] * np.array([w,h,w,h])
            (stratX,startY,endX,endY) = box.astype("int")

            # gambar prediksi di frame
            label = "{}: {:.2f}%".format(CLASSES[idx], 
                confidence * 100)
            cv2.rectangle(frame,(stratX,startY),(endX,endY),COLORS[idx],2) 
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame,label,(startY,y),cv2.FONT_HERSHEY_SIMPLEX,0.5,COLORS[idx],2)
            
    #menampilkan output ke frame
    cv2.imshow("Frame",frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    fps.update()

#fps stop
fps.stop()
print("[INFO] elapsed time : {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
vs.stop()
