import numpy as np
import cv2
import sys
from calibrate import ARUCO_DICT
import argparse
import time
import os
import socket
import threading

# Adapted form GSNCodes ArUCo-Markers-Pose-Estimation-Generation-Python
# https://github.com/GSNCodes/ArUCo-Markers-Pose-Estimation-Generation-Python

# ####################### Socket configurations ############################################ #

# Server configuration
HOST = '127.0.0.1'  # Localhost
PORT = 65432        # Port for the server

# Start Point: X, Y, Z+distance camera roboter TCP, RX, -RY, -RZ
VECTOR_TO_ROBOT = [0, -350, 905, 0.01, 2.216, -2.242]

INTERNAL_CAM_VECTOR = [-32.5, 8, 0, -3.142, 0, 0]

# Shared list to store messages
received_messages = []


def server_thread():
    """Thread function to handle incoming client connections."""
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, PORT))
    server_socket.listen()
    print(f"Server started at {HOST}:{PORT}")

    while True:
        conn, addr = server_socket.accept()
        print(f"Connected by {addr}")
        threading.Thread(target=handle_client, args=(conn, addr), daemon=True).start()


def handle_client(conn, addr):
    """Handles a single client connection."""
    try:
        while True:
            data = conn.recv(1024)
            if not data:
                break
            message = data.decode()
            print(f"Received message: {message}")
            received_messages.append(message)  # Store the message for retrieval
    except ConnectionResetError:
        print(f"Connection with {addr} closed.")
    finally:
        conn.close()


# Example function to retrieve received messages
def get_received_messages():
    """Returns all messages received so far."""
    return received_messages

# ###################### Pose estimation ##################################### #


def calc_transformation_matrix(r_vec, t_vec):
    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(r_vec)

    # Translation vector
    translation_vector = t_vec

    # Construct the transformation matrix
    transformation_matrix = np.hstack((rotation_matrix, translation_vector.reshape(3, 1)))
    transformation_matrix = np.vstack((transformation_matrix, np.array([0, 0, 0, 1])))
    return transformation_matrix


def img_point_to_world_point(r_vec_img, t_vec_img, marker_size, r_vec_world, t_vec_world):
    # Construct the image transformation matrix
    img_trans_matrix = calc_transformation_matrix(r_vec_img, t_vec_img)

    # Define the local coordinates of the marker's corners - Assuming the marker's center is at the origin (0, 0, 0)
    local_points = np.array([
        [-marker_size / 2, -marker_size / 2, 0, 1],
        [marker_size / 2, -marker_size / 2, 0, 1],
        [marker_size / 2, marker_size / 2, 0, 1],
        [-marker_size / 2, marker_size / 2, 0, 1]
    ], dtype=np.float32)

    # Transform the local points to the camera coordinate system
    camera_points = np.dot(img_trans_matrix, local_points.T).T

    # Extract the 3D coordinates
    camera_points = camera_points[:, :3]
    return camera_points


def pose_estimation(frame, aruco_dict_type, camera_coefficients, distortion_coefficients, tag_size,
                    world_trans_vector, world_rot_vector):
    '''
    frame - Frame from the video stream
    matrix_coefficients - Intrinsic matrix of the calibrated camera
    distortion_coefficients - Distortion coefficients associated with your camera
    tag_size - Size of ArUco Tag in pixels
    world_translation_vector - Translation vector of the camera calibration
    world_rotation_vector - Rotation vector of the camera calibration

    return:-
    frame - The frame with the axis drawn on it
    '''

    # Tag size convertion from pixels to millimeters
    metric_tag_size = tag_size*0.2645833333
    # metric_tag_size = tag_size # for output in pixel

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)

    # Initialize the detector parameters using DetectorParameters
    parameters = cv2.aruco.DetectorParameters()

    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, rejected = detector.detectMarkers(gray)

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

    list_coordinates = []
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
            # Determine coordinates of marker's origin and account for internal corrections and rotations
            coordinates = img_point_to_world_point(rvec, tvec, metric_tag_size, world_rot_vector, world_trans_vector)
            origin_x = np.mean(coordinates[:, 0]) + INTERNAL_CAM_VECTOR[0]
            origin_y = np.mean(coordinates[:, 1]) + INTERNAL_CAM_VECTOR[1]
            origin_z = np.mean(coordinates[:, 2]) + INTERNAL_CAM_VECTOR[2]
            origins = list([origin_x, origin_y, origin_z])
            rots = np.squeeze(rvec)
            rots[0] = rots[0] + INTERNAL_CAM_VECTOR[3]
            origins.extend(rots)
            coords_rots = [float(round(elem, 2)) for elem in origins]
            list_coordinates.append(coords_rots)

            text = f"Origin of ArUco marker {i+1}: X = {origin_x:.1f} mm, Y = {origin_y:.1f} mm, Z = {origin_z:.1f} mm"
            (text_width, text_height), text_baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            cv2.putText(frame, text, (10, (1+(i*2))*text_height), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7,
                        (0, 0, 255), 1, cv2.LINE_AA)
            text_2 = f"RX = {rots[0]:.1f} rad, RY = {rots[1]:.1f} rad, RZ = {rots[2]:.1f} rad"
            cv2.putText(frame, text_2, (10, (2+(i*2)) * text_height), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7,
                        (0, 0, 255), 1, cv2.LINE_AA)
            # Check scaling and transformation
            marker_length = np.abs(coordinates[0, 0] - coordinates[1, 0])
            text_marker_size = f"Length of ArUco marker {i+1}: {marker_length:.1f} mm"
            (text_marker_width, text_marker_height), text_baseline = cv2.getTextSize(text_marker_size, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            cv2.putText(frame, text_marker_size,
                        ((frame.shape[1] - (text_marker_width - 10)), (frame.shape[0] - ((i + 1) * text_marker_height + 10))),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (0, 0, 255), 1, cv2.LINE_AA)

    return frame, list_coordinates


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
    ap.add_argument("-i", "--Cam_id", required=False, type=int, default=0, help="Size of tag (pixels)")
    ap.add_argument("-s", "--S_TagSize", required=True, help="Size of tag (px)")
    ap.add_argument("-n", "--Object_name", required=True, help="Name of the test object")
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
    cam_id = args["Cam_id"]
    video_state = eval(args["video"])
    video_dir = args["folder"]
    obj_name = args["Object_name"]

    k = calibration_parameters_path['cam_matrix']
    d = calibration_parameters_path['dist_coeff']
    r = calibration_parameters_path['r_vec']
    t = calibration_parameters_path['t_vec']
    s = int(tag_pixel_size)

    # Setup socket server for message exchange with ur3e roboter control_loop.py
    server_thread_instance = threading.Thread(target=server_thread, daemon=True)
    server_thread_instance.start()

    video = cv2.VideoCapture(cam_id)
    time.sleep(2.0)

    # Variables for saving images while running robot
    save_frame_state = False
    save_frame_pos = ""

    if not os.path.exists("pose_estimation"):
        os.mkdir("pose_estimation")

    if not os.path.exists(f"pose_estimation/{obj_name}"):
        os.mkdir(f"pose_estimation/{obj_name}")

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

        # Check if message from ur3e --> then set save_frame_state=True --> save frame to directory
        if received_messages:
            save_frame_pos = get_received_messages()
            save_frame_pos = eval(save_frame_pos[-1].split('_')[-1])
            save_frame_pos = np.array(save_frame_pos)
            save_frame_pos[:3] = 1000 * save_frame_pos[:3]
            save_frame_pos = np.array(VECTOR_TO_ROBOT) - save_frame_pos
            save_frame_pos = [float(round(coord, 2)) for coord in save_frame_pos]

            save_frame_state = True
            received_messages.clear()  # Clear after processing

        # Perform pose estimation
        output, list_detected_coords = pose_estimation(frame, aruco_dict_type, k, d, s, t, r)

        if video_state:
            # Write the frame to the output files
            video_writer.write(output)

        if len(list_detected_coords) > 0:
            if save_frame_state:
                for nr, marker in enumerate(list_detected_coords):
                    marker = [float(round(coord, 2)) for coord in marker]
                    img_name = f"pose_estimation/{obj_name}/{obj_name}{nr}_{save_frame_pos}_{marker}.jpg"
                    cv2.imwrite(img_name, frame)
            save_frame_state = False

        # draw line for alignment
        width = int(video.get(3))  # float `width`
        height = int(video.get(4))  # float `height`
        cv2.line(output, (int(width/2), 0), (int(width/2), height), (0, 255, 0), 1)
        cv2.line(output, (0, int(height/2)), (width, int(height/2)), (0, 255, 0), 1)
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
