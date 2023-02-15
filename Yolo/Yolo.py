import cv2
import numpy as np

# Yolo load
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes =  [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()

output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]
# colors = np.random.uniform(0, 255, size=(len(classes), 3)) 

#img load
def Yolo(img):
    global top, bottom, left, right
    height, width, channels = img.shape

    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False) # 창 크기 설정
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    # 탐지한 객체의 클래스 예측
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # if confidence > 0.5:
            if class_id == 0 and confidence > 0.5: # 레이블 하나만 적용(person), 원하는 class id 입력(coco.names의 id에서 -1 할 것)
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = "{}: {:.2f}".format(classes[class_ids[i]], confidences[i]*100)
            # colors = colors[i] # 경계 상자 컬러 설정
            colors = [0,255,0] # 단일 색상 사용
            
            if confidences[i]*100 > 95: #사람일 확률이 95%이상일 때
                cv2.rectangle(img, (x - 15, y - 15), (x + w + 15, y + h + 15), colors, 2) # Bounding Box
                # cv2.putText(img, label, (x,y-5), font, 2, colors, 1) # textbox
    return img

num = "141" # 영상 번호
cam = cv2.VideoCapture("sarcopenia/"+ num + "_02" + ".mp4") # 영상경로

while True:
    ret, frame = cam.read()

    if ret:
        if num == "358": # 가로영상
            frame = frame
        # 세로영상    
        elif frame.shape[0] == 1080: # [1080, 1920]
            frame = frame[:,656:1263]
        else:
            frame = frame[:,436:842] # [720, 1280]
        
        detected_img = Yolo(frame)
        cv2.imshow("Image", detected_img)
        # cv2.imwrite("경로" +'.jpg' , detected_img) #image 저장

        if cv2.waitKey(1) & 0xFF == 27: #esc 누르면 종료
            break

    else:
        break