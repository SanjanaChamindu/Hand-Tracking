import cv2
import time
import os
import handTrackingModule as htm

width = 1280
height = 720

cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)


folderPath = "FingerImages"
myList = os.listdir(folderPath)
print(myList)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)

pTime = 0

detector = htm.handDetector(detectionCon=0.75)

tipIds = [4, 8, 12, 16, 20]
currentFinger = 0
previousFinger = 0

def displayFingerCount(img, currentFinger, overlayList, height):
    h, w, c = overlayList[currentFinger - 1].shape
    img[0:h, 0:w] = overlayList[currentFinger - 1]

    rect_start_point = (20, height - 225)
    rect_end_point = (170, height - 25)

    # New coordinates for the text
    text_position = (45, height - 75)

    cv2.rectangle(img, rect_start_point, rect_end_point, (0, 255, 0), cv2.FILLED)
    cv2.putText(img, str(currentFinger), text_position, cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)
    return currentFinger

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    # print(lmList)

    if len(lmList) != 0:
        fingers = []

        # Thumb
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 Fingers
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        currentFinger = fingers.count(1)
    
        displayFingerCount(img, currentFinger, overlayList, height)


    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (500, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 0, 0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)


'''
import cv2
import time
import os
import handTrackingModule as htm

width = 1280
height = 720

cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

folderPath = "FingerImages"
myList = os.listdir(folderPath)
print(myList)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)

pTime = 0

detector = htm.handDetector(detectionCon=0.75)

tipIds = [4, 8, 12, 16, 20]
currentFinger = 0
previousFinger = 0

def displayFingerCount(img, currentFinger, overlayList, height):
    h, w, c = overlayList[currentFinger - 1].shape
    img[0:h, 0:w] = overlayList[currentFinger - 1]

    rect_start_point = (20, height - 225)
    rect_end_point = (170, height - 25)

    # New coordinates for the text
    text_position = (45, height - 75)

    cv2.rectangle(img, rect_start_point, rect_end_point, (0, 255, 0), cv2.FILLED)
    cv2.putText(img, str(currentFinger), text_position, cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)
    return currentFinger

session_count = 0
total_finger_count = 0
session_start_time = time.time()
session_duration = 2  # seconds
gap_duration = 2  # seconds
in_session = True

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    current_time = time.time()

    if in_session and current_time - session_start_time >= session_duration:
        # End of session
        in_session = False
        session_start_time = current_time  # Reset start time for gap

    elif not in_session and current_time - session_start_time >= gap_duration:
        # End of gap, start new session
        in_session = True
        session_start_time = current_time  # Reset start time for session
        session_count += 1

        if session_count >= 10:
            break  # Stop after 10 sessions

    if in_session and len(lmList) != 0:
        fingers = []

        # Thumb
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 Fingers
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        currentFinger = fingers.count(1)
        total_finger_count += currentFinger  # Add current finger count to total
        displayFingerCount(img, currentFinger, overlayList, height)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (500, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

print(f'Total finger count over 10 sessions: {total_finger_count}')
cap.release()
cv2.destroyAllWindows()

'''