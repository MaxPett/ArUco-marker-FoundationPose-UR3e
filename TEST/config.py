# config.py
"""Configuration settings and constants for the camera calibration system."""

import os
from dataclasses import dataclass
import cv2 as cv
import time
from typing import Dict, Union


@dataclass
class CameraConfig:
    CHESS_ROWS = 8
    CHESS_COLS = 6
    FRAME_RATE = 10
    DISPLAY_DURATION_MS = 500

    # Directory setup
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CALIBRATION_DIR = os.path.join(BASE_DIR, "calibration")
    RECORDINGS_DIR = os.path.join(BASE_DIR, "recordings")
    TAGS_DIR = os.path.join(BASE_DIR, "tags")
    CALIBRATION_FILE = os.path.join(BASE_DIR, "calibration.npz")

    # Ensure required directories exist
    @classmethod
    def init_directories(cls):
        for directory in [cls.CALIBRATION_DIR, cls.RECORDINGS_DIR, cls.TAGS_DIR]:
            os.makedirs(directory, exist_ok=True)


class CameraConfig:
    """Configuration settings for the camera calibration system."""

    # Chessboard calibration pattern settings
    CHESS_ROWS: int = 8
    CHESS_COLS: int = 6
    SQUARE_SIZE_MM: float = 25.0  # Size of each square in millimeters

    # Camera settings
    FRAME_WIDTH: int = 1280
    FRAME_HEIGHT: int = 720
    FRAME_RATE: int = 30
    AUTO_FOCUS: int = 1  # 1 for auto, 0 for manual
    EXPOSURE: int = -3  # Negative values for auto exposure

    # Display settings
    DISPLAY_SCALE: float = 1.0
    DISPLAY_DURATION_MS: int = 500
    WINDOW_NAME: str = "Camera Calibration"

    # ArUco marker settings
    ARUCO_DICT_TYPES: Dict[str, int] = {
        "DICT_4X4_50": cv.aruco.DICT_4X4_50,
        "DICT_4X4_100": cv.aruco.DICT_4X4_100,
        "DICT_4X4_250": cv.aruco.DICT_4X4_250,
        "DICT_4X4_1000": cv.aruco.DICT_4X4_1000,
        "DICT_5X5_50": cv.aruco.DICT_5X5_50,
        "DICT_5X5_100": cv.aruco.DICT_5X5_100,
        "DICT_5X5_250": cv.aruco.DICT_5X5_250,
        "DICT_5X5_1000": cv.aruco.DICT_5X5_1000,
        "DICT_6X6_50": cv.aruco.DICT_6X6_50,
        "DICT_6X6_100": cv.aruco.DICT_6X6_100,
        "DICT_6X6_250": cv.aruco.DICT_6X6_250,
        "DICT_6X6_1000": cv.aruco.DICT_6X6_1000,
        "DICT_7X7_50": cv.aruco.DICT_7X7_50,
        "DICT_7X7_100": cv.aruco.DICT_7X7_100,
        "DICT_7X7_250": cv.aruco.DICT_7X7_250,
        "DICT_7X7_1000": cv.aruco.DICT_7X7_1000,
        "DICT_ARUCO_ORIGINAL": cv.aruco.DICT_ARUCO_ORIGINAL,
        "DICT_APRILTAG_16h5": cv.aruco.DICT_APRILTAG_16h5,
        "DICT_APRILTAG_25h9": cv.aruco.DICT_APRILTAG_25h9,
        "DICT_APRILTAG_36h10": cv.aruco.DICT_APRILTAG_36h10,
        "DICT_APRILTAG_36h11": cv.aruco.DICT_APRILTAG_36h11
    }
    DEFAULT_ARUCO_TYPE: str = "DICT_6X6_250"
    MARKER_SIZE_MM: float = 50.0  # Size of ArUco marker in millimeters

    # File and directory settings
    BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
    CALIBRATION_DIR: str = os.path.join(BASE_DIR, "calibration")
    RECORDINGS_DIR: str = os.path.join(BASE_DIR, "recordings")
    TAGS_DIR: str = os.path.join(BASE_DIR, "tags")
    CALIBRATION_FILE: str = os.path.join(BASE_DIR, "calibration.npz")

    # Video recording settings
    VIDEO_CODEC: str = 'XVID'
    VIDEO_FPS: int = 30
    VIDEO_EXTENSION: str = '.avi'

    # Calibration settings
    MIN_CALIBRATION_FRAMES: int = 20
    CALIBRATION_TIMEOUT_SEC: int = 60
    CALIBRATION_DELAY_SEC: float = 2.0  # Delay between captures

    @classmethod
    def init_directories(cls) -> None:
        """Initialize required directories for the application."""
        directories = [cls.CALIBRATION_DIR, cls.RECORDINGS_DIR, cls.TAGS_DIR]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    @classmethod
    def get_camera_properties(cls) -> Dict[int, Union[int, float]]:
        """Return a dictionary of camera properties to be set on initialization."""
        return {
            cv.CAP_PROP_FRAME_WIDTH: cls.FRAME_WIDTH,
            cv.CAP_PROP_FRAME_HEIGHT: cls.FRAME_HEIGHT,
            cv.CAP_PROP_FPS: cls.FRAME_RATE,
            cv.CAP_PROP_AUTOFOCUS: cls.AUTO_FOCUS,
            cv.CAP_PROP_EXPOSURE: cls.EXPOSURE,
        }

    @classmethod
    def get_video_writer_fourcc(cls) -> int:
        """Get the FourCC code for video writing."""
        return cv.VideoWriter_fourcc(*cls.VIDEO_CODEC)

    @classmethod
    def get_aruco_dict(cls, dict_type: str = None) -> cv.aruco.Dictionary:
        """Get the ArUco dictionary based on type string."""
        dict_type = dict_type or cls.DEFAULT_ARUCO_TYPE
        if dict_type not in cls.ARUCO_DICT_TYPES:
            raise ValueError(f"Invalid ArUco dictionary type. Must be one of: {list(cls.ARUCO_DICT_TYPES.keys())}")
        return cv.aruco.getPredefinedDictionary(cls.ARUCO_DICT_TYPES[dict_type])

    @classmethod
    def get_calibration_criteria(cls) -> tuple:
        """Get the calibration termination criteria."""
        return (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    @classmethod
    def get_video_filename(cls, prefix: str = "recording") -> str:
        """Generate a filename for video recording."""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        return os.path.join(cls.RECORDINGS_DIR, f"{prefix}_{timestamp}{cls.VIDEO_EXTENSION}")