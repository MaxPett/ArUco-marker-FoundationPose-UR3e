# Camera Calibration and ArUco Marker Pose Estimation

This project enables camera calibration, video capture, ArUco marker generation, and real-time pose estimation using ArUco markers. It utilizes OpenCV for computer vision tasks and Tkinter for user interface components.

## Features

1. **Camera Calibration**: Calibrate a camera using chessboard patterns for precise vision tasks.
2. **Video Capture and Recording**: Stream and optionally save video from a webcam.
3. **ArUco Marker Generation**: Create ArUco markers for pose estimation.
4. **Real-Time Pose Estimation**: Estimate poses using detected ArUco markers.

## Requirements

- Python 3.x
- OpenCV (`cv2`) and Numpy
- Tkinter (for GUI components)

## Installation

1. Clone the repository:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2. Install dependencies:
    ```bash
    pip install opencv-python-headless numpy
    ```

3. Ensure you have a working camera connected to the system.

## Directory Structure

- **`calibration/`**: Stores images used for camera calibration.
- **`tags/`**: Stores generated ArUco marker images.
- **`Recordings/`**: Saves video recordings (if enabled).

## Usage

1. **Camera Calibration**:
   - Run the script to begin calibration using a chessboard pattern. The calibration images are captured or loaded from `calibration/`.

2. **User Input for Settings**:
   - A GUI will prompt for settings:
     - Whether to save video output.
     - Whether to enable pose estimation.
     - Selection of ArUco marker type and size.

3. **Generate ArUco Marker**:
   - The program generates an ArUco marker with user-specified size and type.

4. **Real-Time Pose Estimation or Video Stream**:
   - If pose estimation is enabled, the program detects the specified ArUco marker and performs real-time pose estimation.
   - Otherwise, it streams the video without pose estimation.

## Example

To start the calibration and pose estimation:
```bash
python main.py
