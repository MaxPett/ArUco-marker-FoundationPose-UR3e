#!/usr/bin/env python

'''
    camera calibration for distorted images with chess board samples
    reads distorted images, calculates the calibration and write undistorted images

    usage:
        calibrate.py [-c <columns>] [-r <rows>] [-t <pattern type>] [-p <path calibration images>] [-n <camera's ID>]
        [--debug <output path>] [--square_size=<square size>] [--marker_size=<aruco marker size>]
        [--aruco_dict=<aruco dictionary name>]

    usage example:
        calibrate.py -c 8 -r 11 -t checkerboard -p calibration -n 0 --square_size=50

    default values:
        --debug:    output/
        -w: 8
        -h: 11
        -t: checkerboard
        -p: calibration
        -n: 0
        --square_size: 20
        --marker_size: 5
        --aruco_dict: DICT_4X4_50
        --threads: 4

    NOTE: Chessboard size is defined in inner corners. Charuco board size is defined in units.
'''

# Python 2/3 compatibility
from __future__ import print_function
import numpy as np
import cv2 as cv
import os
import argparse
import time
from glob import glob

# Define available ArUco tags
ARUCO_DICT = {
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


def splitfn(fn):
    path, fn = os.path.split(fn)
    name, ext = os.path.splitext(fn)
    return path, name, ext


def processImage(fn, pattern_type, pattern_size, pattern_points, charuco_detector, board, w, h, debug_dir):
    print('processing %s... ' % fn)
    img = cv.imread(fn, cv.IMREAD_GRAYSCALE)
    if img is None:
        print("Failed to load", fn)
        return None

    assert w == img.shape[1] and h == img.shape[0], ("size: %d x %d ... " % (img.shape[1], img.shape[0]))
    found = False
    corners = 0
    if pattern_type == 'checkerboard':
        found, corners = cv.findChessboardCorners(img, pattern_size)
        if found:
            term = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 30, 0.001)
            cv.cornerSubPix(img, corners, (5, 5), (-1, -1), term)
            frame_img_points = corners.reshape(-1, 2)
            frame_obj_points = pattern_points
    elif pattern_type == 'charucoboard':
        corners, charucoIds, _, _ = charuco_detector.detectBoard(img)
        if (len(corners) > 0):
            frame_obj_points, frame_img_points = board.matchImagePoints(corners, charucoIds)
            found = True
        else:
            found = False
    else:
        print("unknown pattern type", pattern_type)
        return None

    if not os.path.exists(debug_dir):
        os.mkdir(debug_dir)
    if debug_dir:
        vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        if pattern_type == 'checkerboard':
            cv.drawChessboardCorners(vis, pattern_size, corners, found)
        elif pattern_type == 'charucoboard':
            cv.aruco.drawDetectedCornersCharuco(vis, corners, charucoIds=charucoIds)
        _path, name, _ext = splitfn(fn)
        outfile = os.path.join(debug_dir, name + '_board.png')
        cv.imwrite(outfile, vis)

    if not found:
        print('pattern not found')
        return None

    print('           %s... OK' % fn)
    return (frame_img_points, frame_obj_points)


def stream_video(cam_id, save_dir_calib):
    """
       Streams video from camera and handles calibration image capture.

       Args:
           cam_id (int): Camera device ID
       """
    source = cv.VideoCapture(cam_id)
    win_name = 'Video Stream'

    # Check if the webcam is opened correctly
    if not source.isOpened():
        print("Error: Could not access camera.")
        exit()

    # Setup directories and load calibration if available
    if not os.path.exists(save_dir_calib):
        os.mkdir(save_dir_calib)

    image_counter = 1
    last_capture_time = 0
    display_duration = 1000  # time in milliseconds
    # Stram video loop
    while True:
        # Capture frame-by-frame from the webcam
        has_frame, frame = source.read()
        # If frame was not captured correctly, exit the loop
        if not has_frame:
            print("Error: Failed to capture image.")
            break

        # flip source
        frame = cv.flip(frame, 1)     # flip video
        frame_width = source.get(3)   # float `width`
        frame_height = source.get(4)  # float `height`

        # Check if we need to display the captured image number
        current_time = int(time.time() * 1000)  # Get current time in milliseconds
        if current_time - last_capture_time <= display_duration and image_counter > 1 and not last_capture_time == 0:
            # Display the last captured image number for display_duration time
            text = f"Captured calibration picture {image_counter - 1}"
            (text_width, text_height), text_baseline = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            cv.putText(frame, text, (int(0.5*(frame_width-text_width)), int(0.95*frame_height-0.5*text_height)),
                       cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv.LINE_AA)

        # Display the resulting frame
        cv.imshow(win_name, frame)

        # Capture the key pressed
        key = cv.waitKey(1) & 0xFF
        # If spacebar is pressed, capture and save the image
        if key == 32:  # ASCII for space bar is 32
            img_name = f"calibration_{image_counter}.jpg"
            img_path = os.path.join(save_dir_calib, img_name)
            cv.imwrite(img_path, frame)
            print(f"Image saved as {img_path}")
            image_counter += 1
            last_capture_time = int(time.time() * 1000)  # Reset capture time
        elif key == 113 or key == 27 or cv.getWindowProperty(win_name, cv.WND_PROP_VISIBLE) < 1:
            break

    source.release()
    cv.destroyAllWindows()


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Camera calibration script")

    # Define arguments with their defaults
    parser.add_argument('-c', '--columns', type=int, default=8,
                        help="Number of columns in the chessboard pattern (default: 8)")
    parser.add_argument('-r', '--rows', type=int, default=11,
                        help="Number of rows in the chessboard pattern (default: 11)")
    parser.add_argument('-t', '--type', default='checkerboard', choices=['checkerboard', 'charucoboard'],
                        help="Type of pattern (default: 'checkerboard')")
    parser.add_argument('-p', '--path', type=str, default='calibration', help="Path to calibration images or output path (default: 'calibration')")
    parser.add_argument('-n', '--number', type=int, default=0, help="Camera's ID number (default: 0)")
    parser.add_argument('--debug', default='output', help="Debug output path (default: 'output')")
    parser.add_argument('--square_size', type=float, default=20,
                        help="Size of each square in the checkerboard (default: 20mm)")
    parser.add_argument('--marker_size', type=float, default=5,
                        help="Size of ArUco marker in the pattern (default: 5mm)")
    parser.add_argument('--aruco_dict', default='DICT_4X4_50', help="ArUco's used for charucoboard (default: 'DICT_4X4_50')")
    parser.add_argument('--threads', type=int, default=4, help="Number of threads to use (default: 4)")

    # Parse arguments
    args = parser.parse_args()

    # Extract parsed arguments
    height = args.rows
    width = args.columns
    pattern_type = args.type
    if pattern_type == 'checkerboard':
        height -= 1
        width -= 1
    cam_id = args.number
    square_size = args.square_size
    marker_size = args.marker_size
    aruco_dict_name = args.aruco_dict
    debug_dir = args.debug
    calibration_path = args.path
    threads = args.threads

    img_names = glob(os.path.join(calibration_path, '*.jpg'))
    # Start capturing calibration pictures
    if len(img_names) == 0:
        stream_video(cam_id, calibration_path)  # Capture calibration images if none exist
        img_names = glob(os.path.join(calibration_path, "*.jpg"))

    pattern_size = (width, height)
    if pattern_type == 'checkerboard':
        pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
        pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
        pattern_points *= square_size

    obj_points = []
    img_points = []
    h, w = cv.imread(img_names[0], cv.IMREAD_GRAYSCALE).shape[:2]

    # check charucoboard
    aruco_dicts = ARUCO_DICT
    if aruco_dict_name not in set(aruco_dicts.keys()):
        print("unknown aruco dictionary name")
        return None
    aruco_dict = cv.aruco.getPredefinedDictionary(aruco_dicts[aruco_dict_name])
    board = cv.aruco.CharucoBoard(pattern_size, square_size, marker_size, aruco_dict)
    charuco_detector = cv.aruco.CharucoDetector(board)

    if threads <= 1:
        chessboards = [processImage(fn, pattern_type, pattern_size, pattern_points, charuco_detector, board, w, h, debug_dir) for fn in img_names]
    else:
        print("Run with %d threads..." % threads)
        from multiprocessing.dummy import Pool as ThreadPool
        pool = ThreadPool(threads)
        chessboards = pool.starmap(processImage, [(fn, pattern_type, pattern_size, pattern_points, charuco_detector, board, w, h, debug_dir) for fn in img_names])

    chessboards = [x for x in chessboards if x is not None]
    for (corners, pattern_points) in chessboards:
        img_points.append(corners)
        obj_points.append(pattern_points)

    # calculate camera distortion
    rms, camera_matrix, dist_coefs, rvecs, tvecs = cv.calibrateCamera(obj_points, img_points, (w, h), None, None)

    print("\nReproj Error (pixles): {:.4f}".format(rms))
    print("camera matrix:\n", camera_matrix)
    print("distortion coefficients: ", dist_coefs.ravel())

    # undistort the image with the calibration
    print('')
    for fn in img_names if debug_dir else []:
        _path, name, _ext = splitfn(fn)
        img_found = os.path.join(debug_dir, name + '_board.png')
        outfile = os.path.join(debug_dir, name + '_undistorted.png')

        img = cv.imread(img_found)
        if img is None:
            continue

        h, w = img.shape[:2]
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 1, (w, h))

        # Save calibration parameters
        curr_folder = os.path.dirname(os.path.abspath(__file__))
        param_path = os.path.join(curr_folder, 'calibration.npz')
        np.savez(param_path,
                 cam_matrix=newcameramtx,
                 dist_coeff=dist_coefs,
                 r_vec=rvecs,
                 t_vec=tvecs)

        dst = cv.undistort(img, camera_matrix, dist_coefs, None, newcameramtx)

        # crop and save the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]

        print('Undistorted image written to: %s' % outfile)
        cv.imwrite(outfile, dst)

    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
