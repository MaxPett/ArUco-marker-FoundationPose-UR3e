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
    git clone https://github.com/MaxPett/BSc-MMB
    cd BSc-MMB
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
   - Run the script to begin calibration using a chessboard pattern. The calibration images are captured by pressing the space key when the calibration window is opend or loaded from `calibration/`.
   - number of rows and columns of chessboard needs to reduced by one and inserted in the code
      - Take images in different planes (rotation and translation of the chessboard pattern)
	  - Take multiple images (at least 10 for good results)
      - The entire chessboard should be inside the chessboard	  
   - Window can be closed by pressing ESC or 'q' key or window closed

2. **User Input for Settings**:
   - A GUI will prompt for settings:
     - Whether to save video output.
     - Whether to enable pose estimation.
     - Selection of ArUco marker type and size.

3. **Generate ArUco Marker**:
   - The program generates an ArUco marker with user-specified size and type.

4. **Real-Time Pose Estimation or Video Stream**:
   - If pose estimation is enabled, the program detects the specified ArUco marker (multiple ones possible) and performs real-time pose estimation.
   - Otherwise, it streams the video without pose estimation and shows the undestorted image
   
## Functions Overview
- **cam_calibrate**: Performs camera calibration using chessboard patterns. Displays detected chessboard corners for reviewing purpose.
- **user_requests**: Launches a GUI for user input to configure video saving, pose estimation, ArUco type, and size.
- **video_writer_object**: Creates a video writer object for saving video based on specified file type (.avi or .mp4).
- **load_calibration_from_file**: Loads camera calibration parameters from a saved file.
- **stream_video**: Streams video from a webcam, with options for calibration and saving.
- **run_pose_estimation**: Streams video and performes real-time pose estimation (multiple ArUco objects possible)

## Example

To start the calibration and pose estimation:
```bash
python main.py
