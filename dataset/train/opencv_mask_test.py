import cv2
import numpy as np
import imutils
import time
from matplotlib import pyplot as plt
net = cv2.dnn.readNetFromDarknet("yolov4-custom-mask.cfg","backup/yolov4-custom-mask_2000.weights")
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
classes = [line.strip() for line in open("obj.names")]
colors = [(0,0,255),(255,0,0),(0,255,0)]

def yolo_detect(frame):
    # forward propogation
    img = cv2.resize(frame, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape 
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # get detection boxes
    class_ids = []
    confidences = []
    boxes = []
    
    for out in outs:
        for detection in out:
            tx, ty, tw, th, confidence = detection[0:5]
            scores = detection[5:]
            class_id = np.argmax(scores)  
            if confidence > 0.3:   
                center_x = int(tx * width)
                center_y = int(ty * height)
                w = int(tw * width)
                h = int(th * height)

                # 取得箱子方框座標
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                
    # draw boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return img

# =============
# 讀取圖片做偵測
# =============
# img = cv2.imread("E:/Programming/AI/withMaskDataset/png/785.png")
# im = yolo_detect(img)
# img_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
# plt.imshow(img_rgb)


# =============
# 即時影像偵測
# =============
cap = cv2.VideoCapture("E:/Programming/AI/DataSets/V3/videos/mask_test.mp4")

while True:
    hasFrame, frame = cap.read()
    
    img = yolo_detect(frame)
    cv2.imshow("Frame", imutils.resize(img, width=850))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()