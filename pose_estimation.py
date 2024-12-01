import numpy as np
import cv2
import sys
from calibrate import ARUCO_DICT
import argparse
import time
import os

# Adapted form GSNCodes ArUCo-Markers-Pose-Estimation-Generation-Python
# https://github.com/GSNCodes/ArUCo-Markers-Pose-Estimation-Generation-Python


def pose_estimation(frame, aruco_dict_type, camera_coefficients, distortion_coefficients, tag_size):
    '''
    frame - Frame from the video stream
    matrix_coefficients - Intrinsic matrix of the calibrated camera
    distortion_coefficients - Distortion coefficients associated with your camera
    tag_size - Size of ArUco Tag in pixels

    return:-
    frame - The frame with the axis drawn on it
    '''

    # Tag size convertion from pixels to meters
    metric_tag_size = tag_size*(0.2645833333/1000)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)

    # Initialize the detector parameters using DetectorParameters
    parameters = cv2.aruco.DetectorParameters()

    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, rejected = detector.detectMarkers(gray)

    # Perform subpixel corner detection - might decrease stability or delay detection!
    if len(corners) > 0:
        # Prepare criteria for corner refinement
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
        refined_corners = []
        for marker_corners in corners:
            # Convert to float32 for cornerSubPix
            marker_corners = marker_corners.astype(np.float32)

            # Refine each corner of the marker
            refined_marker_corners = cv2.cornerSubPix(gray, marker_corners, (5, 5), (-1, -1), criteria)
            refined_corners.append(refined_marker_corners)
        # Replace original corners with refined corners
        corners = refined_corners

    # If markers are detected
    if len(corners) > 0 and ids is not None:
        for i in range(len(ids)):
            # Estimate pose of each marker
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i:i+1], metric_tag_size, camera_coefficients,
                                                                distortion_coefficients)
            # Draw markers
            cv2.aruco.drawDetectedMarkers(frame, corners[i:i+1], ids[i:i+1])

            # Draw axis for each marker

            frame = cv2.drawFrameAxes(frame, camera_coefficients, distortion_coefficients,
                                      rvec, tvec, 0.5 * metric_tag_size)

    return frame


def video_writer_object(source, video_name):
    """
        Creates a video writer object for saving video streams.
        Frame resolution, frame dimensions and fourcc code important!

        Args:
            source: Video capture source
            video_name (str): Output video filename

        Returns:
            cv2.VideoWriter: Configured video writer object
        """
    frame_width = int(source.get(3))
    frame_height = int(source.get(4))
    if ".avi" in video_name:
        # Define the codec and create VideoWriter object.
        out_avi = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc("M", "J", "P", "G"), 10, (frame_width, frame_height))
        return out_avi
    else:
        # Define the codec and create VideoWriter object.
        out_mp4 = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*"mp4v"), 10, (frame_width, frame_height))
        return out_mp4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--cam_id", required=False, type=int, default=0, help="Size of tag (pixels)")
    ap.add_argument("-s", "--S_TagSize", required=True, help="Size of tag (mm)")
    ap.add_argument("-c", "--Calib_param", required=True, type=np.load, help="Path to calibration parameters file (numpy file)")
    ap.add_argument("-t", "--type", required=False, type=str, default="DICT_ARUCO_ORIGINAL",
                    help="Type of ArUCo tag to detect")
    ap.add_argument("-v", "--video", required=False, type=str, default="False", help="Video recording (default: False")
    ap.add_argument("-f", "--folder", required=False, type=str, default="recordings",
                    help="Video output directory (default: recordings")
    args = vars(ap.parse_args())

    if ARUCO_DICT.get(args["type"], None) is None:
        print(f"ArUCo tag type '{args['type']}' is not supported")
        sys.exit(0)

    aruco_dict_type = ARUCO_DICT[args["type"]]
    calibration_parameters_path = args["Calib_param"]
    tag_pixel_size = args["S_TagSize"]
    cam_id = args["cam_id"]
    video_state = eval(args["video"])
    video_dir = args["folder"]

    k = calibration_parameters_path['cam_matrix']
    d = calibration_parameters_path['dist_coeff']
    r = calibration_parameters_path['r_vec']
    t = calibration_parameters_path['t_vec']
    s = int(tag_pixel_size)

    video = cv2.VideoCapture(cam_id)
    time.sleep(2.0)

    if video_state:
        if not os.path.exists(video_dir):
            os.mkdir(video)

        timestr = time.strftime("%Y%m%d_%H-%M-%S_")
        file_name = 'pose_estimation_recording.mp4'
        save_path = os.path.join(video_dir, str(timestr + file_name))
        # Initialise videoWriterObject to store video
        video_writer = video_writer_object(video, save_path)

    while True:
        ret, frame = video.read()

        if not ret:
            break

        output = pose_estimation(frame, aruco_dict_type, k, d, s)
        if video_state:
            # Write the frame to the output files
            video_writer.write(output)

        win_name = 'Estimated Pose'
        cv2.imshow(win_name, output)

        key = cv2.waitKey(1) & 0xFF
        if key == 113 or key == 27 or cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    # Cleanup
    if video_state and video_writer:
        print(f"Video {file_name} saved to {save_path}.\n")
        video_writer.release()
    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
