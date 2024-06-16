from ultralytics import YOLO
import cv2
import math 
import numpy as np
import imutils


# start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

def Slope(a,b,c,d):
    return (d - b)/(c - a)

# model
model = YOLO("yolo-Weights/yolov8n.pt")

# object classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]


while True:
    success, img = cap.read()
    results = model(img, stream=True)

    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # class name
            cls = int(box.cls[0])
            print("Class name -->", classNames[cls])

            # put box in cam
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
            print("Confidence --->",confidence)

            # object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)


            if(classNames[cls] == "person"):
                print("Person Coordinates:", (x1, y1), "-", (x2, y2))
                # Crop the person from the image
                person_crop = img[y1:y2, x1:x2]

                # Display only the cropped person in a new window
                cv2.imshow('Cropped Person', person_crop)
                
                # Reading Image
                beltframe = person_crop

                # Resizing The Image
                #beltframe = imutils.resize(beltframe, height=800)

                #Converting To GrayScale
                beltgray = cv2.cvtColor(beltframe, cv2.COLOR_RGB2GRAY)

                # No Belt Detected Yet
                belt = False

                # Bluring The Image For Smoothness
                blur = cv2.blur(beltgray, (1, 1))

                # Converting Image To Edges
                edges = cv2.Canny(blur, 50, 400)


                # Previous Line Slope
                ps = 0

                # Previous Line Co-ordinates
                px1, py1, px2, py2 = 0, 0, 0, 0

                # Extracting Lines
                lines = cv2.HoughLinesP(edges, 1, np.pi/270, 30, maxLineGap = 20, minLineLength = 170)

                # If "lines" Is Not Empty
                if lines is not None:

                    # Loop line by line
                    for line in lines:

                        # Co-ordinates Of Current Line
                        x1, y1, x2, y2 = line[0]

                        # Slope Of Current Line
                        s = Slope(x1,y1,x2,y2)
                        
                        # If Current Line's Slope Is Greater Than 0.7 And Less Than 2
                        if ((abs(s) > 0.7) and (abs (s) < 2)):

                            # And Previous Line's Slope Is Within 0.7 To 2
                            if((abs(ps) > 0.7) and (abs(ps) < 2)):

                                # And Both The Lines Are Not Too Far From Each Other
                                if(((abs(x1 - px1) > 5) and (abs(x2 - px2) > 5)) or ((abs(y1 - py1) > 5) and (abs(y2 - py2) > 5))):

                                    # Plot The Lines On "beltframe"
                                    cv2.line(beltframe, (x1, y1), (x2, y2), (0, 0, 255), 3)
                                    cv2.line(beltframe, (px1, py1), (px2, py2), (0, 0, 255), 3)

                                    # Belt Is Detected
                                    print ("Belt Detected")
                                    belt = True

                        # Otherwise Current Slope Becomes Previous Slope (ps) And Current Line Becomes Previous Line (px1, py1, px2, py2)            
                        ps = s
                        px1, py1, px2, py2 = line[0]
                        
                                
                if belt == False:
                    print("No Seatbelt detected")

                # Show The "beltframe"
                cv2.imshow("Image Processing", edges)


    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
