import cv2
import time
import os

# Create folders
os.makedirs("captured/body", exist_ok=True)
os.makedirs("captured/face", exist_ok=True)

# Load camera
cap = cv2.VideoCapture(0)

# Load HOG human detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Load face detector
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

captured = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect humans
    humans, _ = hog.detectMultiScale(
        frame,
        winStride=(8, 8),
        padding=(16, 16),
        scale=1.05
    )

    for (x, y, w, h) in humans:
        if captured:
            continue

        # Draw human box
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

        person_roi = frame[y:y+h, x:x+w]
        gray_roi = gray[y:y+h, x:x+w]

        # Detect face inside person
        faces = face_cascade.detectMultiScale(
            gray_roi,
            scaleFactor=1.3,
            minNeighbors=5
        )

        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Save full body
        cv2.imwrite(f"captured/body/person_{timestamp}.jpg", person_roi)

        # Save face(s)
        for (fx, fy, fw, fh) in faces:
            face_img = person_roi[fy:fy+fh, fx:fx+fw]
            cv2.imwrite(f"captured/face/face_{timestamp}.jpg", face_img)

        print("Human detected – Body and face captured")
        captured = True

    cv2.imshow("Human Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
