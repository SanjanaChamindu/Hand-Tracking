import cv2
import time
import handTrackingModule as htm
import joblib
import numpy as np

# Load the model from the file
model = joblib.load('model.pkl')
print("Model loaded from model.pkl")

def main():
    cap = cv2.VideoCapture(0)
    detector = htm.handDetector(detectionCon=0.75)

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)

        if len(lmList) != 0:
            landmarks = [coord for lm in lmList for coord in lm[1:]]
            landmarks = np.array(landmarks).reshape(1, -1)

            # Predict the finger count using the trained model
            predicted_finger_count = model.predict(landmarks)

            # Display the predicted finger count
            cv2.putText(img, f'Finger Count: {predicted_finger_count}', (10, 150), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)



        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
