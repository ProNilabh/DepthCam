import cv2
import time

# Initialize camera 1
cam1 = cv2.VideoCapture(0)
time.sleep(1)  # Small delay to ensure proper initialization

# Initialize camera 2
cam2 = cv2.VideoCapture(1)
time.sleep(1)  # Small delay to ensure proper initialization

# Check if cameras are opened correctly
if not cam1.isOpened():
    print("Error: Could not open camera 1")
    exit()
else:
    print("Camera 1 opened successfully")

if not cam2.isOpened():
    print("Error: Could not open camera 2")
    exit()
else:
    print("Camera 2 opened successfully")

# Set the resolution of the cameras
width = 640
height = 480
cam1.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cam1.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cam2.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cam2.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

print("Press 'q' to exit the camera view")

while True:
    # Capture images from both cameras
    ret1, frame1 = cam1.read()
    ret2, frame2 = cam2.read()

    if not ret1:
        print("Error capturing image from camera 1")
        break

    if not ret2:
        print("Error capturing image from camera 2")
        break

    # Display the images from both cameras
    cv2.imshow('Camera 1', frame1)
    cv2.imshow('Camera 2', frame2)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting on user request")
        break

# Release the cameras and close all windows
cam1.release()
cam2.release()
cv2.destroyAllWindows()
print("Cameras released and windows closed")
