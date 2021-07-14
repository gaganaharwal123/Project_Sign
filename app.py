
from flask import Flask, render_template, Response
import cv2

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from collections import deque
import numpy as np
import cv2
import pickle
import random
import math as m

app = Flask(__name__)


def gen_frames():  
    mlp_model = load_model('emnist_mlp_model.h5')
    # cnn_model = load_model('emnist_cnn_model.h5')

    letters = { 1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h', 9: 'i', 10: 'j',
    11: 'k', 12: 'l', 13: 'm', 14: 'n', 15: 'o', 16: 'p', 17: 'q', 18: 'r', 19: 's', 20: 't',
    21: 'u', 22: 'v', 23: 'w', 24: 'x', 25: 'y', 26: 'z', 27: '-'}

    blueLower = np.array([100, 60, 60])
    blueUpper = np.array([140, 255, 255])

    kernel = np.ones((5, 5), np.uint8)

    blackboard = np.zeros((480,640,3), dtype=np.uint8)
    alphabet = np.zeros((200, 200, 3), dtype=np.uint8)

    points = deque(maxlen=512)

    prediction1 = 26
    prediction2 = 26

    index = 0
    camera = cv2.VideoCapture(0)
    
    while True:
        (grabbed, frame) = camera.read()
        frame = cv2.flip(frame, 1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        blueMask = cv2.inRange(hsv, blueLower, blueUpper)
        blueMask = cv2.erode(blueMask, kernel, iterations=2)
        blueMask = cv2.morphologyEx(blueMask, cv2.MORPH_OPEN, kernel)
        blueMask = cv2.dilate(blueMask, kernel, iterations=1)

        (_, cnts, _) = cv2.findContours(blueMask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        center = None

        if len(cnts) > 0:
            cnt = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
            ((x, y), radius) = cv2.minEnclosingCircle(cnt)
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            M = cv2.moments(cnt)
            center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

            points.appendleft(center)

        elif len(cnts) == 0:
            if len(points) != 0:
                blackboard_gray = cv2.cvtColor(blackboard, cv2.COLOR_BGR2GRAY)
                blur1 = cv2.medianBlur(blackboard_gray, 15)
                blur1 = cv2.GaussianBlur(blur1, (5, 5), 0)
                thresh1 = cv2.threshold(blur1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                blackboard_cnts = cv2.findContours(thresh1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]
                if len(blackboard_cnts) >= 1:
                    cnt = sorted(blackboard_cnts, key = cv2.contourArea, reverse = True)[0]

                    if cv2.contourArea(cnt) > 1000:
                        x, y, w, h = cv2.boundingRect(cnt)
                        alphabet = blackboard_gray[y-10:y + h + 10, x-10:x + w + 10]
                        newImage = cv2.resize(alphabet, (28, 28))
                        newImage = np.array(newImage)
                        newImage = newImage.astype('float32')/255

                        prediction1 = mlp_model.predict(newImage.reshape(1,28,28))[0]
                        prediction1 = np.argmax(prediction1)

                        # prediction2 = cnn_model.predict(newImage.reshape(1,28,28,1))[0]
                        # prediction2 = np.argmax(prediction2)

                points = deque(maxlen=512)
                blackboard = np.zeros((480, 640, 3), dtype=np.uint8)

        for i in range(1, len(points)):
                if points[i - 1] is None or points[i] is None:
                        continue
                cv2.line(frame, points[i - 1], points[i], (0, 0, 0), 2)
                cv2.line(blackboard, points[i - 1], points[i], (255, 255, 255), 8)

        cv2.putText(frame, "Convolution Neural Network predected word:  " + str(letters[int(prediction2)+1]), (10, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("alphabets Recognition Real Time", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()
@app.route('/')
def index():
    return render_template('templates/index.html')
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__ == '__main__':
    app.run(debug=False)
# def sign_create():


#     if(ch == 1):
        
#         def nothing(x):
#             pass

#         cv2.namedWindow('image')

#         cv2.createTrackbar('HMin','image',0,179,nothing)
#         cv2.createTrackbar('SMin','image',0,255,nothing)
#         cv2.createTrackbar('VMin','image',0,255,nothing)
#         cv2.createTrackbar('HMax','image',0,179,nothing)
#         cv2.createTrackbar('SMax','image',0,255,nothing)
#         cv2.createTrackbar('VMax','image',0,255,nothing)

#         cv2.setTrackbarPos('HMax', 'image', 179)
#         cv2.setTrackbarPos('SMax', 'image', 255)
#         cv2.setTrackbarPos('VMax', 'image', 255)

#         hMin = sMin = vMin = hMax = sMax = vMax = 0
#         phMin = psMin = pvMin = phMax = psMax = pvMax = 0

#         cap = cv2.VideoCapture(0)

#         waitTime = 330

#         while(1):

#             ret, img = cap.read()
#             output = img

#             hMin = cv2.getTrackbarPos('HMin','image')
#             sMin = cv2.getTrackbarPos('SMin','image')
#             vMin = cv2.getTrackbarPos('VMin','image')

#             hMax = cv2.getTrackbarPos('HMax','image')
#             sMax = cv2.getTrackbarPos('SMax','image')
#             vMax = cv2.getTrackbarPos('VMax','image')

#             lower = np.array([hMin, sMin, vMin])
#             upper = np.array([hMax, sMax, vMax])

#             hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#             mask = cv2.inRange(hsv, lower, upper)
#             output = cv2.bitwise_and(img,img, mask= mask)

#             if( (phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax) ):
#                 phMin = hMin
#                 psMin = sMin
#                 pvMin = vMin
#                 phMax = hMax
#                 psMax = sMax
#                 pvMax = vMax
#                 old_values = {"hMin": hMin, "sMin": sMin, "vMin": vMin, "hMax": hMax, "sMax": sMax, "vMax": vMax}  
                
#             pickle.dump(old_values, open("old_values.p", "wb"))  

#             output = cv2.flip(output,1)
#             cv2.imshow('image',output)
            
#             k = cv2.waitKey(5) & 0xFF
#             if k == 27:
#                 break

#         cap.release()
#         cv2.destroyAllWindows()

#     else:
        
#         old_values = pickle.load(open("old_values.p", "rb"))



#     n=1
#     while(n==1):
        
            
#         cap = cv2.VideoCapture(0)
        
#         center_points = deque()
        
#         while True:
        
#             _, frame = cap.read()
#             frame = cv2.flip(frame, 1)

#             blur_frame = cv2.GaussianBlur(frame, (7, 7), 0)

#             hsv = cv2.cvtColor(blur_frame, cv2.COLOR_BGR2HSV)

#             lower_blue = np.array([old_values.get('hMin'), old_values.get('sMin'), old_values.get('vMin')])
#             upper_blue = np.array([old_values.get('hMax'), old_values.get('sMax'), old_values.get('vMax')])
#             mask = cv2.inRange(hsv, lower_blue, upper_blue)

#             kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

#             mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

#             contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2:]

#             if len(contours) > 0:
                
#                 biggest_contour = max(contours, key=cv2.contourArea)

#                 moments = cv2.moments(biggest_contour)
#                 centre_of_contour = (int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00']))
#                 cv2.circle(frame, centre_of_contour, 5, (0, 0, 255), -1)

#                 ellipse = cv2.fitEllipse(biggest_contour)
#                 cv2.ellipse(frame, ellipse, (0, 255, 255), 2)

#                 center_points.appendleft(centre_of_contour)

#             for i in range(1, len(center_points)):
#                 b = random.randint(230, 255)
#                 g = random.randint(100, 255)
#                 r = random.randint(100, 255)
#                 if m.sqrt(((center_points[i - 1][0] - center_points[i][0]) ** 2) + (
#                         (center_points[i - 1][1] - center_points[i][1]) ** 2)) <= 50:
#                     cv2.line(frame, center_points[i - 1], center_points[i], (b, g, r), 4)
#                     cv2.line(mask, center_points[i - 1], center_points[i], (b, g, r), 4)

#             cv2.imshow('Original', frame)
#             cv2.imshow('Mask', mask)

#             filename = "outputs/sign_%d.jpg"%ocount
#             cv2.imwrite(filename, mask)

#             k = cv2.waitKey(5) & 0xFF
#             if k == 27:
#                 break
#             if k == 113:
#                 ocount+=1
#                 pickle.dump(ocount, open("ocount.p", "wb"))
#                 n = 2
#                 break

        
#         cv2.destroyAllWindows()
#         cap.release()
