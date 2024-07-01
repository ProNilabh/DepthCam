import cv2
import numpy as np

# Define the dimensions of the chessboard
chessboard_size = (9, 6)

# Load an image
img = cv2.imread(r'C:\Users\NESAC\Desktop\Nilabh\9x6_1-8cm_chessboard.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Find chessboard corners
ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

# If found, draw corners
if ret:
    cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
    cv2.imshow('Chessboard Corners', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Chessboard corners not detected")
