"""Main entry point for the camera calibration application."""

import cv2 as cv
from tkinter import messagebox
import os
import numpy as np

# Local imports
from config import CameraConfig
from camera import Camera
from calibration import CameraCalibrator
from gui import SettingsDialog, UserPreferences


def generate_aruco_tag(output_path, tag_id, tag_type, tag_size):
    """
    Generates an ArUCo tag and saves it to the specified output path.

    Parameters:
    - output_path (str): Path to save the generated ArUCo tag image.
    - tag_id (int): ID of the ArUCo tag to generate.
    - tag_type (str): Type of ArUCo tag to generate.
    - tag_size (int): Size of the ArUCo tag in pixels.
    """

    ARUCO_DICT = CameraConfig.ARUCO_DICT_TYPES
    # Check if the dictionary type is supported
    if ARUCO_DICT.get(tag_type, None) is None:
        print(f"ArUCo tag type '{tag_type}' is not supported")
        return

    # Get the dictionary and generate the tag
    arucoDict = cv.aruco.getPredefinedDictionary(ARUCO_DICT[tag_type])
    print(f"Generating ArUCo tag of type '{tag_type}' with ID '{tag_id}'")

    # Generate the ArUco tag
    tag = np.zeros((tag_size, tag_size, 1), dtype="uint8")
    arucoDict.generateImageMarker(tag_id, tag_size, tag, 1)

    # Save the generated tag
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    tag_name = f"{output_path}/{tag_type}_id_{tag_id}.png"
    cv.imwrite(tag_name, tag)
    print(f"ArUCo tag saved to {tag_name}")

    # Display the tag
    cv.imshow("ArUCo Tag", tag)
    while True:
        key = cv.waitKey(1) & 0xFF
        if key in [27, ord('q')]:  # ESC or 'q' key
            break
        if cv.getWindowProperty("ArUCo Tag", cv.WND_PROP_VISIBLE) < 1:
            break
    cv.destroyAllWindows()


def run_video_stream(camera: Camera, settings: UserPreferences):
    """Handle basic video streaming functionality."""
    while True:
        success, frame = camera.get_frame()
        if not success:
            break

        if settings.save_video:
            # Add video saving logic here
            pass

        cv.imshow('Video Stream', frame)

        # Break loop on 'q' press
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    camera.stop()


def run_pose_estimation(camera: Camera, settings: UserPreferences):
    """Handle pose estimation with ArUco markers."""
    # Load calibration parameters
    calibration_data = CameraCalibrator.load_calibration()
    if not calibration_data:
        raise RuntimeError("Camera must be calibrated before running pose estimation")

    camera_matrix, dist_coeffs = calibration_data
    ARUCO_DICT = CameraConfig.ARUCO_DICT_TYPES
    # Initialize ArUco detector based on settings.aruco_type
    aruco_dict = cv.aruco.getPredefinedDictionary(ARUCO_DICT[settings.aruco_type])
    parameters = cv.aruco.DetectorParameters()
    detector = cv.aruco.ArucoDetector(aruco_dict, parameters)

    while True:
        success, frame = camera.get_frame()
        if not success:
            break

        # Detect ArUco markers
        corners, ids, rejected = detector.detectMarkers(frame)

        if ids is not None:
            # Draw detected markers
            cv.aruco.drawDetectedMarkers(frame, corners, ids)

            # Estimate pose for each marker
            rvecs, tvecs, _ = cv.aruco.estimatePoseSingleMarkers(
                corners,
                settings.video_size,  # marker size in real-world units
                camera_matrix,
                dist_coeffs
            )

            # Draw axes for each detected marker
            for i in range(len(ids)):
                cv.drawFrameAxes(frame, camera_matrix, dist_coeffs,
                                 rvecs[i], tvecs[i], settings.video_size / 2)

        cv.imshow('Pose Estimation', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    camera.stop()


def main():
    # Initialize configuration
    CameraConfig.init_directories()

    try:
        # Set up camera and calibrator
        camera = Camera()
        camera.start()
        calibrator = CameraCalibrator(CameraConfig())

        # Get user preferences
        settings = SettingsDialog().get_settings()
        if not settings:
            return

        # Perform calibration if needed
        if not CameraCalibrator.load_calibration():
            camera_matrix, dist_coeffs = calibrator.calibrate()

        # Generate the ArUCo tag
        aruco_tag_output_dir = CameraConfig.TAGS_DIR
        aruco_tag_type = settings.aruco_type
        aruco_tag_id = CameraConfig.ARUCO_DICT_TYPES[aruco_tag_type]
        aruco_tag_size = settings.video_size
        generate_aruco_tag(aruco_tag_output_dir, aruco_tag_id, aruco_tag_type, aruco_tag_size)

        # Run the appropriate mode based on user selection
        if settings.run_pose_estimation:
            run_pose_estimation(camera, settings)
        else:
            run_video_stream(camera, settings)

    except Exception as e:
        messagebox.showerror("Error", str(e))
    finally:
        cv.destroyAllWindows()
        if camera:
            camera.stop()


if __name__ == "__main__":
    main()