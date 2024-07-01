import cv2

# Initialize camera 1
cam1 = cv2.VideoCapture(0)

# Check if camera 1 is opened correctly
if not cam1.isOpened():
    print("Error: Could not open camera 1")
    exit()
else:
    print("Camera 1 opened successfully")

# Set the resolution of the camera
width = 640
height = 480
cam1.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cam1.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

print("Press 'q' to exit the camera view")
while True:
    ret1, frame1 = cam1.read()
    if not ret1:
        print("Error capturing image from camera 1")
        break

    # Display the image from camera 1
    cv2.imshow('Camera 1', frame1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting camera 1 view on user request")
        break

# Release the camera and close all windows
cam1.release()
cv2.destroyAllWindows()
print("Camera 1 released and window closed")
