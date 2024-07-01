import cv2
import numpy as np

# Load calibration data
calibration_data = np.load('calibration_data.npz')
mtx_l = calibration_data['mtx_l']
dist_l = calibration_data['dist_l']
mtx_r = calibration_data['mtx_r']
dist_r = calibration_data['dist_r']
R = calibration_data['R']
T = calibration_data['T']
R1 = calibration_data['R1']
R2 = calibration_data['R2']
P1 = calibration_data['P1']
P2 = calibration_data['P2']
Q = calibration_data['Q']

# Create VideoCapture objects for both cameras
cap_l = cv2.VideoCapture(0)
cap_r = cv2.VideoCapture(1)

# Check if cameras opened successfully
if not cap_l.isOpened() or not cap_r.isOpened():
    print("Error: Could not open one or both cameras.")
    exit()

# Set up the StereoSGBM parameters
min_disp = 0
num_disp = 160  # Needs to be divisible by 16
block_size = 5
stereo = cv2.StereoSGBM_create(minDisparity=min_disp, numDisparities=num_disp, blockSize=block_size)

try:
    while True:
        # Capture frames
        ret_l, frame_l = cap_l.read()
        ret_r, frame_r = cap_r.read()

        if not ret_l or not ret_r:
            print("Failed to capture frame.")
            break

        # Rotate the frames 90 degrees counterclockwise
        frame_l = cv2.rotate(frame_l, cv2.ROTATE_90_COUNTERCLOCKWISE)
        frame_r = cv2.rotate(frame_r, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Rectify images
        map_l1, map_l2 = cv2.initUndistortRectifyMap(mtx_l, dist_l, R1, P1, frame_l.shape[1::-1], cv2.CV_16SC2)
        map_r1, map_r2 = cv2.initUndistortRectifyMap(mtx_r, dist_r, R2, P2, frame_r.shape[1::-1], cv2.CV_16SC2)

        rectified_l = cv2.remap(frame_l, map_l1, map_l2, cv2.INTER_LINEAR)
        rectified_r = cv2.remap(frame_r, map_r1, map_r2, cv2.INTER_LINEAR)

        # Compute disparity map
        disparity = stereo.compute(rectified_l, rectified_r).astype(np.float32) / 16.0
        print("Min Disparity:", disparity.min(), "Max Disparity:", disparity.max())

        if Q is not None:
            # Generate depth map
            depth_map = cv2.reprojectImageTo3D(disparity, Q)
            mask = disparity > disparity.min()  # Mask for valid disparity values

            if mask.any():
                # Find the depth at the center
                center_x, center_y = depth_map.shape[1] // 2, depth_map.shape[0] // 2
                depth_at_center = depth_map[center_y, center_x]
                if depth_at_center[2] != 0:  # Check if the depth value is valid
                    distance = np.linalg.norm(depth_at_center)  # Euclidean distance from the camera
                    print(f"Distance to object: {distance * 100:.2f} cm")
                else:
                    print("Invalid depth value at the center pixel.")
            else:
                print("No valid disparity values detected.")

        # Display images
        cv2.imshow('Left Camera', rectified_l)
        cv2.imshow('Right Camera', rectified_r)
        cv2.imshow('Disparity', disparity / disparity.max())

        if Q is not None:
            cv2.imshow('Depth Map', depth_map)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap_l.release()
    cap_r.release()
    cv2.destroyAllWindows()
