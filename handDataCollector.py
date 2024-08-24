import cv2
import handTrackingModule as htm
import time

def collect_hand_data(expected_finger_count, iterations):
    cap = cv2.VideoCapture(0)
    detector = htm.handDetector(detectionCon=0.75)
    collected_data = []

    iteration_count = 0
    while iteration_count < iterations:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)

        if len(lmList) != 0:
            # Flatten the landmark positions into a single list
            landmarks = [coord for lm in lmList for coord in lm[1:]]
            # Append the landmark data and expected finger count to the collected data list
            collected_data.append((landmarks, expected_finger_count))
            iteration_count += 1

            # Display the hand gesture and the number of iterations collected
            cv2.putText(img, f'Iteration: {iteration_count}/{iterations}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)
            cv2.putText(img, f'Expected Finger Count: {expected_finger_count}', (10, 110), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Add a small delay to ensure the window stays responsive
        time.sleep(0.1)

    cap.release()
    cv2.destroyAllWindows()

    return collected_data

def get_finger_data():
    # Get user input for expected finger count and number of iterations
    finger_data = []
    while True:
        expected_finger_count = int(input("Enter the expected finger count: "))
        if expected_finger_count < 0:
            break

        iterations = int(input("Enter the number of iterations: "))
        if iterations < 0:
            break

        # Collect hand data
        data = collect_hand_data(expected_finger_count, iterations)
        finger_data.extend(data)

    return finger_data



if __name__ == "__main__":
    finger_data = get_finger_data()
    print(len(finger_data))