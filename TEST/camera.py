# camera.py
"""Camera handling and video streaming functionality."""

import cv2 as cv
import os
import time
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Callable
from config import CameraConfig


@dataclass
class CameraState:
    image_counter: int = 1
    last_capture_time: int = 0


class Camera:
    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self.capture = None
        self.state = CameraState()
        self.config = CameraConfig()
        self.running = False

    def start(self) -> bool:
        """Initialize camera capture."""
        self.capture = cv.VideoCapture(self.camera_id)
        if not self.capture.isOpened():
            raise RuntimeError(f"Failed to open camera {self.camera_id}")
        self.running = True
        return True

    def stop(self):
        """Release camera resources."""
        self.running = False
        if self.capture:
            self.capture.release()
            self.capture = None

    def get_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Capture a frame from the camera."""
        if not self.capture:
            return False, None
        success, frame = self.capture.read()
        if success:
            return True, cv.flip(frame, 1)  # Flip horizontally
        return False, None

    def save_calibration_image(self, frame: np.ndarray) -> str:
        """Save a calibration image and update state."""
        img_name = f"calibration_{self.state.image_counter}.jpg"
        img_path = os.path.join(self.config.CALIBRATION_DIR, img_name)

        if cv.imwrite(img_path, frame):
            print(f"Saved calibration image {self.state.image_counter}")
            self.state.image_counter += 1
            self.state.last_capture_time = int(time.time() * 1000)
            return img_path
        raise RuntimeError(f"Failed to save calibration image: {img_path}")

    def capture_calibration_images(self) -> bool:
        """Run calibration image capture session."""
        if not self.start():
            return False

        window_name = 'Calibration Image Capture'
        cv.namedWindow(window_name)

        try:
            while self.running:
                success, frame = self.get_frame()
                if not success:
                    break

                # Add overlay text for user instructions
                text = "Press SPACE to capture calibration image, Q to finish"
                cv.putText(frame, text, (10, 30), cv.FONT_HERSHEY_SIMPLEX,
                           0.7, (0, 255, 0), 2)

                # Show capture confirmation if recent capture
                if time.time() * 1000 - self.state.last_capture_time < self.config.DISPLAY_DURATION_MS:
                    confirm_text = f"Captured image {self.state.image_counter - 1}"
                    cv.putText(frame, confirm_text, (10, 60), cv.FONT_HERSHEY_SIMPLEX,
                               0.7, (0, 0, 255), 2)

                cv.imshow(window_name, frame)

                key = cv.waitKey(1) & 0xFF
                if key == ord('q') or cv.getWindowProperty(window_name, cv.WND_PROP_VISIBLE) < 1:
                    break
                elif key == ord(' '):  # Spacebar
                    self.save_calibration_image(frame)

        finally:
            self.stop()
            cv.destroyWindow(window_name)

        return self.state.image_counter > 1  # Return True if at least one image was captured
