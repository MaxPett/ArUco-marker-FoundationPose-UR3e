import numpy as np
import cv2
import sys
from calibrate import ARUCO_DICT
import argparse
import time


# Adapted form GSNCodes ArUCo-Markers-Pose-Estimation-Generation-Python
# https://github.com/GSNCodes/ArUCo-Markers-Pose-Estimation-Generation-Python

def pose_estimation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients, tag_size):
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

    # If markers are detected
    if len(corners) > 0 and ids is not None:
        for i in range(len(ids)):
            # Estimate pose of each marker
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i:i+1], metric_tag_size, matrix_coefficients,
                                                                distortion_coefficients)
            # Draw markers
            cv2.aruco.drawDetectedMarkers(frame, corners[i:i+1], ids[i:i+1])

            # Draw axis for each marker

            frame = cv2.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients,
                                      rvec, tvec, 0.5*metric_tag_size)

    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--S_TagSize", required=True, help="Size of tag (pixels)")
    ap.add_argument("-k", "--K_Matrix", required=True, help="Path to calibration matrix (numpy file)")
    ap.add_argument("-d", "--D_Coeff", required=True, help="Path to distortion coefficients (numpy file)")
    ap.add_argument("-t", "--type", type=str, default="DICT_ARUCO_ORIGINAL", help="Type of ArUCo tag to detect")
    args = vars(ap.parse_args())

    if ARUCO_DICT.get(args["type"], None) is None:
        print(f"ArUCo tag type '{args['type']}' is not supported")
        sys.exit(0)

    aruco_dict_type = ARUCO_DICT[args["type"]]
    calibration_matrix_path = args["K_Matrix"]
    distortion_coefficients_path = args["D_Coeff"]
    tag_pixel_size = args["S_TagSize"]

    k = np.load(calibration_matrix_path)
    d = np.load(distortion_coefficients_path)
    s = int(tag_pixel_size)

    video = cv2.VideoCapture(0)
    time.sleep(2.0)

    while True:
        ret, frame = video.read()

        if not ret:
            break

        output = pose_estimation(frame, aruco_dict_type, k, d, s)
        cv2.imshow('Estimated Pose Main', output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
