import math

import Core.Core as ie
import threading
import time
import cv2
import csv
import Object.SSDDetection as sd
import numpy as np

def initProcess(frequency, model,csv):
    core = ie.Core(model)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open the camera.")
    else:
        thread = threading.Thread(target=launchProcess(frequency, core, cap,csv))
        thread.start()
    return thread


def launchProcess(frequency, core, cap,csv):
    processTime = 1
    while True:
        initTime = time.time()
        ret, frame = cap.read()
        if ret:
            height, width, _ = frame.shape
            core.inputImage(frame)
            core.infer()
            data = core.getData(0)
            confidence = core.getData(1)
            boxes = decodeAnchors(data,confidence,csv)
            box = onlyHighest(boxes,confidence,0.5)
            if box != None:
                drawBox(frame,box)
            frame = printFPS(processTime, frame)
            cv2.imshow("Test", frame)
        else:
            print("Error: Couldn't capture an image from the camera.")
        remaining_sleep_time = 1 / frequency - (time.time() - initTime)
        if remaining_sleep_time > 0:
            time.sleep(remaining_sleep_time)
        processTime = (time.time() - initTime) * 1000
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def printFPS(processTime, cap):
    position = (50, 50)  # (x, y) coordinates
    # Define the font, size, and color
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (0, 0, 255)  # BGR color (red in this example)
    # Add the text to the image
    cv2.putText(cap, f"Process Time : {processTime:.2f} ms | FPS : {1000 / processTime:.2f}", position, font, font_scale,
                font_color, thickness=2)
    return cap


def decodeAnchors(data,confidence,csv_file):
    csv_values = csvReader(csv_file)
    list = []
    confidence = confidence/192
    for index in range(len(csv_values)):
        row = csv_values[index]
        value = data[0][index]/192
        sx = row[0]
        sy = row[1]
        w = row[2]
        h = row[3]
        cx = sx + value[0]
        cy = sy + value[1]
        h = h * value[2]
        w = w * value[3]
        x = cx - w * 0.5
        y = cy - h * 0.5
        keypoints = []
        for i in range(7):
            keypoints.append((cx+value[4+i*2],cy+value[4+i*2+1]))
        list.append(sd.SSDDetection(x,y,h,w,center=(cx,cy),keypoints=keypoints,confidence=sigmoid(confidence[0][index])))
    return list

def drawBox(frame,box):
    height, width, _ = frame.shape
    top_left = (int(box.getX() * width), int(box.getY() * height))
    bottom_right = (int(box.getX2() * width), int(box.getY2() * height))

    # Define the color and thickness of the rectangle (in BGR format)
    color = (0, 0, 255)  # Red color in BGR
    thickness = 2

    # Draw the rectangle on the image
    cv2.rectangle(frame, top_left, bottom_right, color, thickness)
    cv2.circle(frame, (int(box.getCenter()[0] * width), int(box.getCenter()[1] * height)), 3, (0, 255, 0), 10)
    #cv2.circle(frame, top_left, 3, (0, 255, 0), 10)
    #cv2.circle(frame, bottom_right, 3, (0, 255, 0), 10)
    for keypoint in box.getKeypoints():
        cv2.circle(frame, (int(keypoint[0] * width), int(keypoint[1] * height)), 3, (255, 0, 0), 10)

def csvReader(csv_file):
    matrix = []
    with open(csv_file, 'r') as file:
        csv_reader = csv.reader(file)

        # Skip the header row if it exists
        header = next(csv_reader, None)

        # Iterate through the rows in the CSV
        for row in csv_reader:
            # Extract values for x, y, h, and w from the row
            x, y, h, w = map(float, row)
            matrix.append([x,y,h,w])

    return matrix

def onlyHighest(boxes,confidence,tresh):
    index_of_max = np.argmax(confidence[0])
    if(confidence[0][index_of_max]<tresh):
        return None
    return boxes[index_of_max]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))