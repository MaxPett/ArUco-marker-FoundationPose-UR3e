# main.py
"""Main entry point for the camera calibration application."""

import cv2 as cv
from tkinter import messagebox

# Local imports
from config import CameraConfig
from camera import Camera
from calibration import CameraCalibrator
from gui import SettingsDialog, UserPreferences


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

    # Initialize ArUco detector based on settings.aruco_type
    aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)
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
        camera.start()  # Add explicit camera start
        calibrator = CameraCalibrator(CameraConfig())

        # Get user preferences
        settings = SettingsDialog().get_settings()
        if not settings:
            return

        # Perform calibration if needed
        if not CameraCalibrator.load_calibration():
            camera_matrix, dist_coeffs = calibrator.calibrate()

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