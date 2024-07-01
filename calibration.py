import cv2
import numpy as np
import glob

# Define the dimensions of the checkerboard
CHECKERBOARD = (6, 9)  # Number of inner corners per chessboard row and column
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d points in real world space
imgpoints_l = []  # 2d points in image plane for left images
imgpoints_r = []  # 2d points in image plane for right images

# Prepare object points based on the checkerboard pattern
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# Load images
left_images = glob.glob('left_images/*.jpg')  # Adjust the path to your left images
right_images = glob.glob('right_images/*.jpg')  # Adjust the path to your right images

if not left_images or not right_images:
    print("Error: No images found in the specified directories.")
    exit()

for left_img_path, right_img_path in zip(left_images, right_images):
    img_l = cv2.imread(left_img_path)
    img_r = cv2.imread(right_img_path)

    if img_l is None or img_r is None:
        print(f"Error: Could not read one or both images: {left_img_path}, {right_img_path}")
        continue

    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret_l, corners_l = cv2.findChessboardCorners(gray_l, CHECKERBOARD, None)
    ret_r, corners_r = cv2.findChessboardCorners(gray_r, CHECKERBOARD, None)

    if ret_l and ret_r:
        objpoints.append(objp)
        imgpoints_l.append(corners_l)
        imgpoints_r.append(corners_r)

        cv2.drawChessboardCorners(img_l, CHECKERBOARD, corners_l, ret_l)
        cv2.drawChessboardCorners(img_r, CHECKERBOARD, corners_r, ret_r)
        cv2.imshow('Left Image', img_l)
        cv2.imshow('Right Image', img_r)
        cv2.waitKey(500)  # Display images for half a second
    else:
        print(f"Checkerboard corners not found in images: {left_img_path}, {right_img_path}")

cv2.destroyAllWindows()

if not objpoints or not imgpoints_l or not imgpoints_r:
    print("Error: Not enough corner points detected for calibration.")
    exit()

print("Starting camera calibration...")

# Calibrate the left and right cameras
ret_l, mtx_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(
    objpoints, imgpoints_l, gray_l.shape[::-1], None, None)
ret_r, mtx_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(
    objpoints, imgpoints_r, gray_r.shape[::-1], None, None)

if not ret_l or not ret_r:
    print("Error: Calibration failed for one or both cameras.")
    exit()

print("Stereo calibration...")

# Stereo calibration
flags = cv2.CALIB_FIX_INTRINSIC  # Fix the intrinsic camera parameters
ret, mtx_l, dist_l, mtx_r, dist_r, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_l, imgpoints_r,
    mtx_l, dist_l, mtx_r, dist_r,
    gray_l.shape[::-1], criteria=criteria, flags=flags)

if not ret:
    print("Error: Stereo calibration failed.")
    exit()

print("Rectifying cameras...")

# Rectify the cameras
R1, R2, P1, P2, Q = cv2.stereoRectify(mtx_l, dist_l, mtx_r, dist_r, gray_l.shape[::-1], R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1)[:5]

print("Saving calibration data...")

# Save calibration data
np.savez('calibration_data.npz', mtx_l=mtx_l, dist_l=dist_l, mtx_r=mtx_r, dist_r=dist_r, R=R, T=T, E=E, F=F, R1=R1, R2=R2, P1=P1, P2=P2, Q=Q)

print("Calibration is complete and data has been saved.")
