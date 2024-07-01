import cv2
import numpy as np

# Paths to the images
left_image_path = r'C:\Users\NESAC\Desktop\Nilabh\left\image1.jpg'  # Replace with your actual path
right_image_path = r'C:\Users\NESAC\Desktop\Nilabh\right\image1.jpg'  # Replace with your actual path

# Load images
img_l = cv2.imread(left_image_path)
img_r = cv2.imread(right_image_path)

# Check if images are loaded
if img_l is None:
    print(f"Error: Unable to load image from {left_image_path}")
if img_r is None:
    print(f"Error: Unable to load image from {right_image_path}")

# Display loaded images
if img_l is not None and img_r is not None:
    cv2.imshow('Left Image', img_l)
    cv2.imshow('Right Image', img_r)
    cv2.waitKey(0)  # Wait for a key press to proceed
    cv2.destroyAllWindows()

# If images are loaded correctly, proceed with rectification
if img_l is not None and img_r is not None:
    # Get image size
    h, w = img_l.shape[:2]

    # Load the calibration data
    with np.load('calibration_data.npz') as data:
        mtx_l = data['mtx_l']
        dist_l = data['dist_l']
        mtx_r = data['mtx_r']
        dist_r = data['dist_r']
        R = data['R']
        T = data['T']

    # Rectification
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        mtx_l, dist_l, mtx_r, dist_r, (w, h), R, T, alpha=0
    )
    map1_l, map2_l = cv2.initUndistortRectifyMap(mtx_l, dist_l, R1, P1, (w, h), cv2.CV_16SC2)
    map1_r, map2_r = cv2.initUndistortRectifyMap(mtx_r, dist_r, R2, P2, (w, h), cv2.CV_16SC2)

    rectified_l = cv2.remap(img_l, map1_l, map2_l, cv2.INTER_LINEAR)
    rectified_r = cv2.remap(img_r, map1_r, map2_r, cv2.INTER_LINEAR)

    # Display rectified images
    cv2.imshow('Rectified Left', rectified_l)
    cv2.imshow('Rectified Right', rectified_r)
    cv2.waitKey(0)  # Wait for a key press to proceed
    cv2.destroyAllWindows()

    # Create stereo matcher
    stereo = cv2.StereoBM_create(numDisparities=16*5, blockSize=15)
    disparity = stereo.compute(cv2.cvtColor(rectified_l, cv2.COLOR_BGR2GRAY), cv2.cvtColor(rectified_r, cv2.COLOR_BGR2GRAY))

    # Normalize the disparity for visualization
    disparity_normalized = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    disparity_normalized = np.uint8(disparity_normalized)

    # Show the disparity map
    cv2.imshow('Disparity', disparity_normalized)
    cv2.waitKey(0)  # Wait for a key press to proceed
    cv2.destroyAllWindows()

    # Calculate and display depth
    depth = cv2.reprojectImageTo3D(disparity, Q)
    distance = np.mean(depth[:, :, 2])  # Average distance
    print(f"Approximate distance: {distance:.2f} cm")

    # Save disparity and depth maps for debugging
    cv2.imwrite('disparity_map.jpg', disparity_normalized)
    np.save('depth_map.npy', depth)

else:
    print("One or both images could not be loaded. Check the file paths.")
