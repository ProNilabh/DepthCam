import cv2
import time
import os

# Camera indices (adjust if necessary)
left_camera_index = 0
right_camera_index = 1

# Create directories for storing images
left_dir = "left_images"
right_dir = "right_images"
os.makedirs(left_dir, exist_ok=True)
os.makedirs(right_dir, exist_ok=True)

# Open both cameras
cap_left = cv2.VideoCapture(left_camera_index)
cap_right = cv2.VideoCapture(right_camera_index)

# Check if cameras opened successfully
if not cap_left.isOpened() or not cap_right.isOpened():
    print("Error: Could not open one or both cameras.")
    exit()

# Display the initial camera feeds for adjustment
print("Position yourself. Camera feeds will be shown for 5 seconds.")
start_time = time.time()
while time.time() - start_time < 5:
    ret_left, frame_left = cap_left.read()
    ret_right, frame_right = cap_right.read()

    if not ret_left or not ret_right:
        print("Failed to capture images from one or both cameras.")
        continue

    # Rotate the frames 90 degrees to the left
    frame_left = cv2.rotate(frame_left, cv2.ROTATE_90_COUNTERCLOCKWISE)
    frame_right = cv2.rotate(frame_right, cv2.ROTATE_90_COUNTERCLOCKWISE)

    cv2.imshow("Left Camera - Position Yourself", frame_left)
    cv2.imshow("Right Camera - Position Yourself", frame_right)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Give a 5-second head start
print("Starting in 5 seconds...")
time.sleep(5)

# Capture 100 images at 3-second intervals
num_images = 100
for i in range(num_images):
    # Countdown before each capture
    print(f"Capturing image {i+1}/{num_images} in 3 seconds...")
    for countdown in range(3, 0, -1):
        print(countdown)
        time.sleep(1)

    # Read frames from both cameras
    ret_left, frame_left = cap_left.read()
    ret_right, frame_right = cap_right.read()

    if not ret_left or not ret_right:
        print("Failed to capture images from one or both cameras.")
        continue

    # Rotate the frames 90 degrees to the left
    frame_left = cv2.rotate(frame_left, cv2.ROTATE_90_COUNTERCLOCKWISE)
    frame_right = cv2.rotate(frame_right, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Display the frames
    cv2.imshow("Left Camera", frame_left)
    cv2.imshow("Right Camera", frame_right)

    # Save images
    left_filename = os.path.join(left_dir, f"image{i+1}.jpg")
    right_filename = os.path.join(right_dir, f"image{i+1}.jpg")
    cv2.imwrite(left_filename, frame_left)
    cv2.imwrite(right_filename, frame_right)
    print(f"Images {i+1} captured and saved.")

    # Wait for a key press and exit early if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Wait for the next capture
    time.sleep(3)

# Release the cameras and close windows
cap_left.release()
cap_right.release()
cv2.destroyAllWindows()

print("Capture completed.")
