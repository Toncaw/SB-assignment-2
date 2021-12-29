import cv2
import numpy as np
import glob
import random
import imageio
import glob
import cv2
import imageio
from numpy import expand_dims

# Load Yolo
#net = cv2.dnn.readNet("yolov3_training_true.weights", "yolov3_testing.cfg")
net = cv2.dnn.readNet("yolov3_training_true.weights", "yolov3_testing.cfg")

#net = cv2.dnn.readNet("yolov4_training_true.weights", "yolov4-cars.cfg")
#net = cv2.dnn.readNet("yolov4_training_last.weights", "yolov4-cars.cfg")

# Name custom object
classes = ["Ear"]

# Images path
images_path = glob.glob(r"test\*.png")

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

class DetectedBoxes:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h


def calc_tp_tn_fp_fn(rectangles, reference_image, indeksi):
    f = reference_image.replace('png','txt')
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    steviloKvadratkov = 0

    for i in range(0, len(indeksi)):
        with open(f) as file:
            lines = file.readlines()
            lines = [line.rstrip() for line in lines]
            for line in lines:
                line = line.split(" ")
                x = int(int(line[1])-int(line[3])/2)
                y = int(int(line[2])-int(line[4])/2)
                w = int(line[3])
                h = int(line[4])

                print(rectangles[i])
                # if not (x < rectangles[i][0] - rectangles[i][2] or x > rectangles[i][0] + rectangles[i][2] and y < rectangles[i][1] - rectangles[i][3] or y > rectangles[i][1] + rectangles[i][3]):
                #     print("true: ", x, y, w, h)
                #     steviloKvadratkov += 1

            xp = rectangles[i][0];
            yp = rectangles[i][1];
            wp = rectangles[i][2];
            hp = rectangles[i][3];

            for x1 in range(480):
                for y1 in range(360):
                    if((x1 in range(x, x+w) and y1 in range(y, y+h)) and (x1 in range(xp, xp+wp) and y1 in range(yp, yp+hp))):
                        tp += 1
                    elif((x1 not in range(x, x+w) and y1 not in range(y, y+h)) and (x1 not in range(xp, xp+wp) and y1 not in range(yp, yp+hp))):
                        tn += 1
                    elif ((x1 not in range(x, x + w) and y1 not in range(y, y + h)) and (x1 in range(xp, xp + wp) and y1 in range(yp, yp + hp))):
                        fp += 1
                    elif ((x1 in range(x, x + w) and y1 in range(y, y + h)) and (x1 not in range(xp, xp + wp) and y1 not in range(yp, yp + hp))):
                        fn += 1

    print(reference_image, " ", tp, tn, fp, fn)
    return tp, tn, fp, fn

def showDetection():
    # Insert here the path of your images
    random.shuffle(images_path)
    # loop through all the images
    for img_path in images_path:

       # img_path = 'test/0017.png'

        # Loading image
        img = cv2.imread(img_path)
        img = cv2.resize(img, None, fx=0.4, fy=0.4)
        height, width, channels = img.shape

        # Detecting objects
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        net.setInput(blob)
        outs = net.forward(output_layers)

        # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.3:
                    # Object detected
                    print(class_id)
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        print("indeksi: ", indexes)

        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = colors[class_ids[i]]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                #cv2.putText(img, label, (x, y + 30), font, 3, color, 2)


        cv2.imshow("Image", img)
        key = cv2.waitKey(0)

    cv2.destroyAllWindows()


def detect_ears():
    accuracies = []
    precisions = []
    recalls = []
    IoUs = []
    for img_path in images_path:

        #img_path = 'test/0017.png'

        img = cv2.imread(img_path)
        height, width, channels = img.shape

        # Detecting objects
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        net.setInput(blob)
        outs = net.forward(output_layers)

        # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.3:
                    # Object detected
                    # print(class_id)
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        tp, tn, fp, fn = calc_tp_tn_fp_fn(boxes, img_path,indexes)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        accuracies.append(accuracy)
        if tp + fp is not 0:
            precision = tp / (tp + fp)
        else:
            precision = 0
        precisions.append(precision)
        if tp + fn is not 0:
            recall = tp / (tp + fn)
        else:
            recall = 0
        recalls.append(recall)
        Iou = tp / (tp + fn + fp)
        IoUs.append(Iou)
        print("Image: " + img_path + " Accuracy: " + str(accuracy) + " Precision: " + str(precision) + " Recall: " + str(recall) + " IoU: " + str(Iou))
    return accuracies, precisions, recalls, IoUs

if __name__ == '__main__':
   # showDetection()

    f_accuracies, f_precisions, f_recalls, f_IoUs = detect_ears()
    a = np.asarray(f_accuracies)
    p = np.asarray(f_precisions)
    r = np.asarray(f_recalls)
    i = np.asarray(f_IoUs)
    print("Accuracy mean:" + str(a.mean()) + " st.dev.: "+str(a.std()))
    print("Precision mean:" + str(p.mean()) + " st.dev.: "+str(p.std()))
    print("Recall mean:" + str(r.mean()) + " st.dev.: "+str(r.std()))
    print("IoU mean:" + str(i.mean()) + " st.dev.: "+str(i.std()))
