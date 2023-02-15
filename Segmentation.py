from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2
# Apply the transformations needed
import torchvision.transforms as T

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

classes = []
with open("coco.names", "r") as f:
    classes =  [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()

output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]

# YOLO
def Yolo(img):
    height, width, channels = img.shape

    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if class_id == 0 and confidence > 0.5: #레이블 하나만 적용
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
            color = [0,255,255]
            # no = 0
            if float(label[-5:]) > 95: # 사람일 확률이 95%이상
                top = y
                bottom = y+h
                left = x
                right = x+w
                # cv2.rectangle(img, (x, y), (x + w , y + h), color, 2) # Bounding Box
            
            else: # YOLO로 사람을 인식을 못 했을경우
                top = "Error"
                bottom = "Error"
                left = "Error"
                right = "Error"
                
    return top, bottom, left, right

# Segmentation
def decode_segmap(image, source, nc=21):
    
    label_colors = np.array([(0, 0, 0),  # 0=background
                    # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                    (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0),
                    # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                    (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0),
                    # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                    (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (192, 128, 128),
                    # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                    (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)])

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
   
    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]
    # rgb map
    rgb = np.stack([r, g, b], axis=2)

    # Load the foreground input image 
    foreground = source

    # Change the color of foreground image to RGB 
    # and resize image to match shape of R-band in RGB output map
    foreground = cv2.resize(foreground,(r.shape[1],r.shape[0]))

    # Create a background array to hold black pixels
    # with the same size as RGB output map
    background = np.ones_like(rgb).astype(np.uint8) # black

    # Convert uint8 to float
    foreground = foreground.astype(float)
    background = background.astype(float)

    # Create a binary mask of the RGB output map using the threshold value 0
    th, alpha = cv2.threshold(np.array(rgb), 0,255, cv2.THRESH_BINARY)

    # Apply a slight blur to the mask to soften edges
    alpha = cv2.GaussianBlur(alpha, (7,7), 0)

    # Normalize the alpha mask to keep intensity between 0 and 1
    alpha = alpha.astype(float)/255

    # Multiply the foreground with the alpha matte
    foreground = cv2.multiply(alpha, foreground)  
  
    # Multiply the background with ( 1 - alpha )
    background = cv2.multiply(1.0 - alpha, background)  
  
    # Add the masked foreground and background
    outImage = cv2.add(foreground, background)

    # Return a normalized output image for display
    return outImage/255 # 0 ~ 1사이 값

def segment(net, image, show_orig=True, dev='cuda'):
    img =  Image.fromarray(np.uint8(image)) 

    if show_orig: cv2.imshow(img); cv2.waitKey(0)
    
    # Comment the Resize and CenterCrop for better inference results
    trf = T.Compose([T.Resize(406), 
                   #T.CenterCrop(224), 
                   T.ToTensor(), 
                   T.Normalize(mean = [0.485, 0.456, 0.406], 
                               std = [0.229, 0.224, 0.225])])

    inp = trf(img).unsqueeze(0).to(dev)
    out = net.to(dev)(inp)['out'][0] 
    # label별 예측 값 [label(21), image.shape[0], image.shpae[1]] 
    # [[0=background에 대한 예측값],[1=aeroplane에 대한 예측값] ... [19=train에 대한 예측값],[20=tv/monitor에 대한 예측값]]

    # top, bottom, left, right, top_no, bottom_no, left_no, right_no = Yolo(image)
    top, bottom, left, right = Yolo(image)
    om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()  

    if not top == "Error":
        om_copy = np.copy(om)
        error_range = 15
        for h in range(om.shape[0]):
            for w in range(om.shape[1]):
                if om[h][w] == 15:
                    if ((top-error_range) < h < (bottom+error_range)) and ((left-error_range) < w < (right+error_range)):
                        om_copy[h-10:h+10, w-10:w+10] = 15
                    else:
                        om_copy[h,w] = 0
    else:
        om_copy = np.copy(om)
        for h in range(om.shape[0]):
            for w in range(om.shape[1]):
                if om[h][w] == 15:
                    om_copy[h-10:h+10, w-10:w+10] = 15

    rgb = decode_segmap(om_copy, image)
    
    return rgb
  

dlab = models.segmentation.deeplabv3_resnet101(pretrained=1).eval()

# VIDEO
num = "141" # 영상 번호
cam = cv2.VideoCapture("yolo/sarcopenia/sarcopenia/"+ num + "_02" + ".mp4") # 영상경로
    
n = 0
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
            
        seg = segment(dlab, frame, show_orig=False)
        # cv2.imwrite("./201_02/%03d.jpg" % n , seg*255) #저장할 때 이미지 픽셀 값이 0~255 사이값이여야함
        cv2.imshow("segmentation", seg)
        n += 1

        if cv2.waitKey(1) & 0xFF == 27: #esc 누르면 종료
            break

    else:
        break