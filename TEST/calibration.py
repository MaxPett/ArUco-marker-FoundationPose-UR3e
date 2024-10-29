# calibration.py
"""Camera calibration functionality."""

import cv2 as cv
import numpy as np
import os
import glob
from typing import Tuple, List, Optional
from config import CameraConfig
from camera import Camera


class CameraCalibrator:
    def __init__(self, config: CameraConfig):
        self.config = config

    def calibrate(self, show_results: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Perform camera calibration using saved images."""
        # Get calibration images
        images = glob.glob(os.path.join(self.config.CALIBRATION_DIR, "*.jpg"))

        # If no images found, start capture session
        if not images:
            print("No calibration images found. Starting capture session...")
            camera = Camera()
            if not camera.capture_calibration_images():
                raise RuntimeError("No calibration images were captured")
            images = glob.glob(os.path.join(self.config.CALIBRATION_DIR, "*.jpg"))

        # Initialize calibration parameters
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        world_points = np.zeros((self.config.CHESS_ROWS * self.config.CHESS_COLS, 3), np.float32)
        world_points[:, :2] = np.mgrid[0:self.config.CHESS_ROWS, 0:self.config.CHESS_COLS].T.reshape(-1, 2)

        world_points_list = []
        image_points_list = []

        # Process each calibration image
        for img_path in images:
            img = cv.imread(img_path)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            found, corners = cv.findChessboardCorners(gray,
                                                      (self.config.CHESS_ROWS, self.config.CHESS_COLS),
                                                      None)

            if found:
                world_points_list.append(world_points)
                refined_corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                image_points_list.append(refined_corners)

                if show_results:
                    cv.drawChessboardCorners(img,
                                             (self.config.CHESS_ROWS, self.config.CHESS_COLS),
                                             refined_corners,
                                             found)
                    cv.imshow('Chessboard Detection', img)
                    cv.waitKey(500)

        cv.destroyAllWindows()

        if not world_points_list:
            raise RuntimeError("No valid chessboard patterns found in calibration images")

        # Perform calibration
        ret, cam_matrix, dist_coeff, rvecs, tvecs = cv.calibrateCamera(
            world_points_list,
            image_points_list,
            gray.shape[::-1],
            None,
            None
        )

        # Save calibration parameters
        np.savez(self.config.CALIBRATION_FILE,
                 ret=ret,
                 camera_matrix=cam_matrix,
                 dist_coeff=dist_coeff,
                 rvecs=rvecs,
                 tvecs=tvecs)

        print(f"Camera calibration completed with reprojection error: {ret:.4f}")
        return cam_matrix, dist_coeff

    @classmethod
    def load_calibration(cls) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Load existing calibration parameters."""
        try:
            data = np.load(CameraConfig.CALIBRATION_FILE)
            return data['camera_matrix'], data['dist_coeff']
        except (FileNotFoundError, IOError):
            return None
