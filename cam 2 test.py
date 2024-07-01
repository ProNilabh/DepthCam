import cv2

# Initialize camera 2
cam2 = cv2.VideoCapture(1)

# Check if camera 2 is opened correctly
if not cam2.isOpened():
    print("Error: Could not open camera 2")
    exit()
else:
    print("Camera 2 opened successfully")

# Set the resolution of the camera
width = 640
height = 480
cam2.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cam2.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

print("Press 'q' to exit the camera view")
while True:
    ret2, frame2 = cam2.read()
    if not ret2:
        print("Error capturing image from camera 2")
        break

    # Display the image from camera 2
    cv2.imshow('Camera 2', frame2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting camera 2 view on user request")
        break

# Release the camera and close all windows
cam2.release()
cv2.destroyAllWindows()
print("Camera 2 released and window closed")
