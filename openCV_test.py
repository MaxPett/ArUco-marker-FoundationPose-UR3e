import cv2 as cv
import glob
import time
import tkinter as tk
from tkinter import ttk, messagebox
import os
import numpy as np
from generate_aruco_tags import generate_aruco_tag
from pose_estimation import pose_estimation
from utils import ARUCO_DICT




"""
Main script for camera calibration and ArUco marker pose estimation.
This program provides functionality for:
1. Camera calibration using a chessboard pattern
2. Video capture and recording
3. ArUco marker generation
4. Real-time pose estimation using ArUco markers
"""

# Global variables for managing calibration image capture
IMAGE_COUNTER = 1  # Tracks the number of calibration images taken
LAST_CAPTURE_TIME = 0  # Timestamp of last image capture
DISPLAY_DURATION = 500  # Duration (ms) to display capture confirmation


def cam_calibrate(showPics=True):
    """
        Performs camera calibration using chessboard pattern images and is based on the code of @ Kevin Wood

        Args:
            showPics (bool): If True, displays detected chessboard corners during calibration

        Returns:
            tuple: Camera matrix and distortion coefficients
        """
    # Read calibration images or capture new ones if none exist
    root = os.getcwd()
    calibration_dir = os.path.join(root, "calibration")
    list_img_paths = glob.glob(os.path.join(calibration_dir, "*.jpg"))

    # Start capturing calibration pictures
    if len(list_img_paths) == 0:
        stream_video(0, False)  # Capture calibration images if none exist
        list_img_paths = glob.glob(os.path.join(calibration_dir, "*.jpg"))

    # Define chessboard parameters
    n_rows = 8
    m_cols = 6
    term_crit = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Initialize arrays for world and image points
    world_points_cur = np.zeros((n_rows*m_cols, 3), np.float32)
    world_points_cur[:, :2] = np.mgrid[0:n_rows, 0:m_cols].T.reshape(-1, 2)
    list_world_points = []
    list_img_points = []

    # Process each calibration image
    for cur_img_path in list_img_paths:
        img_BGR = cv.imread(cur_img_path)
        img_gray = cv.cvtColor(img_BGR, cv.COLOR_BGR2GRAY)
        cor_found, cor_org = cv.findChessboardCorners(img_gray, (n_rows, m_cols), cv.CALIB_CB_ADAPTIVE_THRESH)

        if cor_found:
            list_world_points.append(world_points_cur)
            # Refine corner detection for better accuracy
            cor_refined = cv.cornerSubPix(img_gray, cor_org, (5, 5), (-1, -1), term_crit)
            list_img_points.append(cor_refined)
            if showPics:  # Display detected corners if requested
                cv.drawChessboardCorners(img_BGR, (n_rows, m_cols), cor_refined, cor_found)
                cv.imshow('Webcam - Chessboard', img_BGR)
                cv.waitKey(1000)
    cv.destroyAllWindows()

    # Perform camera calibration
    rep_err, cam_matrix, dist_coeff, r_vec, t_vec = cv.calibrateCamera(list_world_points, list_img_points,
                                                                       img_gray.shape[::-1], None, None)
    print('Camera Matrix:\n', cam_matrix)
    print("Reproj Error (pixles): {:.4f}".format(rep_err))

    # Save calibration parameters
    curr_folder = os.path.dirname(os.path.abspath(__file__))
    param_path = os.path.join(curr_folder, 'calibration.npz')
    np.savez(param_path,
             rep_err=rep_err,
             cam_matrix=cam_matrix,
             dist_coeff=dist_coeff,
             r_vec=r_vec,
             t_vec=t_vec)

    return cam_matrix, dist_coeff


def user_requests():
    """
        Creates a GUI window for user input on video settings.

        Returns:
            tuple: (save_video, run_pose_estimation, aruco_type, video_size)
        """
    # Create a new window
    new_window = tk.Tk()
    new_window.title("Video Settings")

    # Set the size of the window
    new_window.geometry("500x500+650+250")

    # GUI control variables
    save_result = tk.BooleanVar()
    pose_result = tk.BooleanVar()
    aruco_type = tk.StringVar()
    video_size = tk.StringVar(value="200")
    save_yes_var = tk.BooleanVar()
    save_no_var = tk.BooleanVar()
    pose_yes_var = tk.BooleanVar()
    pose_no_var = tk.BooleanVar()

    # Input validation for ArUco marker size
    def validate_size(value):
        if value == "": return True  # Allow empty field for typing
        try:
            size = int(value)
            return size > 0  # Only allow positive integers
        except ValueError:
            return False

    validate_cmd = new_window.register(validate_size)

    # Checkbox handlers to ensure mutual exclusivity
    def on_save_yes():
        if save_yes_var.get():
            save_no_var.set(False)

    def on_save_no():
        if save_no_var.get():
            save_yes_var.set(False)

    def on_pose_yes():
        if pose_yes_var.get():
            pose_no_var.set(False)

    def on_pose_no():
        if pose_no_var.get():
            pose_yes_var.set(False)

    # Form submission handler
    def on_submit():
        # Validate all inputs before proceeding
        if not (save_yes_var.get() or save_no_var.get()):
            messagebox.showerror("Error", "Please select Yes or No for saving video")
            return
        if not (pose_yes_var.get() or pose_no_var.get()):
            messagebox.showerror("Error", "Please select Yes or No for pose estimation")
            return
        try:
            size = int(video_size.get())
            if size <= 0:
                messagebox.showerror("Error", "Please enter a positive number for size")
                return
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number for size")
            return

        save_result.set(save_yes_var.get())  # True if Yes is checked, False if No is checked
        pose_result.set(pose_yes_var.get())  # True if Yes is checked, False if No is checked
        new_window.destroy()

    # GUI Layout setup
    main_frame = ttk.Frame(new_window, padding="20")
    main_frame.pack(fill=tk.BOTH, expand=True)

    # Save video question
    save_label = ttk.Label(main_frame, text="Do you want to save the following video?", font=("Arial", 12))
    save_label.pack(pady=10)

    save_frame = ttk.Frame(main_frame)
    save_frame.pack(pady=5)
    save_yes_check = ttk.Checkbutton(save_frame, text="Yes", variable=save_yes_var, command=on_save_yes)
    save_yes_check.pack(side=tk.LEFT, padx=10)
    save_no_check = ttk.Checkbutton(save_frame, text="No", variable=save_no_var, command=on_save_no)
    save_no_check.pack(side=tk.LEFT, padx=10)

    # Pose estimation question
    pose_label = ttk.Label(main_frame, text="Do you want to run pose estimation?", font=("Arial", 12))
    pose_label.pack(pady=10)

    pose_frame = ttk.Frame(main_frame)
    pose_frame.pack(pady=5)
    pose_yes_check = ttk.Checkbutton(pose_frame, text="Yes", variable=pose_yes_var, command=on_pose_yes)
    pose_yes_check.pack(side=tk.LEFT, padx=10)
    pose_no_check = ttk.Checkbutton(pose_frame, text="No", variable=pose_no_var, command=on_pose_no)
    pose_no_check.pack(side=tk.LEFT, padx=10)

    # ArUco dictionary dropdown
    aruco_label = ttk.Label(main_frame, text="Select ArUco Dictionary:", font=("Arial", 12))
    aruco_label.pack(pady=10)
    aruco_dropdown = ttk.Combobox(main_frame, textvariable=aruco_type, values=list(ARUCO_DICT.keys()), state="readonly")
    aruco_dropdown.pack()
    aruco_dropdown.set(list(ARUCO_DICT.keys())[0])  # Set default value

    # ArUco size input
    size_frame = ttk.Frame(main_frame)
    size_frame.pack(pady=10)

    size_label = ttk.Label(size_frame, text="Enter ArUco size (pixels):", font=("Arial", 12))
    size_label.pack(side=tk.LEFT, padx=5)

    size_entry = ttk.Entry(size_frame, textvariable=video_size, width=10,
                           validate="key", validatecommand=(validate_cmd, '%P'))
    size_entry.pack(side=tk.LEFT, padx=5)

    # Submit button
    submit_button = ttk.Button(main_frame, text="Submit", command=on_submit)
    submit_button.pack(pady=20)

    # Wait for the window to close
    new_window.mainloop()

    # Return the results as a tuple (save_video, run_pose_estimation, aruco_type, video_size)
    return save_result.get(), pose_result.get(), aruco_type.get(), int(video_size.get())


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
        out_avi = cv.VideoWriter(video_name, cv.VideoWriter_fourcc("M", "J", "P", "G"), 10, (frame_width, frame_height))
        return out_avi
    else:
        # Define the codec and create VideoWriter object.
        out_mp4 = cv.VideoWriter(video_name, cv.VideoWriter_fourcc(*"mp4v"), 10, (frame_width, frame_height))
        return out_mp4


def load_calibration_from_file(file_path):
    # Load the camera calibration file
    with np.load(file_path) as data:
        # Access each parameter
        rep_err = data['rep_err']
        cam_matrix = data['cam_matrix']
        dist_coeff = data['dist_coeff']
        r_vec = data['r_vec']
        t_vec = data['t_vec']
    return rep_err, cam_matrix, dist_coeff, r_vec, t_vec


def stream_video(cam_id, save_state):
    """
       Streams video from camera and handles calibration image capture.

       Args:
           cam_id (int): Camera device ID
           save_state (bool): Whether to save the video stream
       """
    source = cv.VideoCapture(cam_id)
    win_name = 'Webcam Feed'

    # Check if the webcam is opened correctly
    if not source.isOpened():
        print("Error: Could not open webcam.")
        exit()

    # Setup directories and load calibration if available
    save_dir_calib = "calibration"
    if not os.path.exists(save_dir_calib):
        os.mkdir(save_dir_calib)

    # Initialize camera calibration if available
    calib_file_path = "calibration.npz"
    if os.path.exists(calib_file_path):
        rep_err, cam_matrix, dist_coeff, r_vec, t_vec = load_calibration_from_file(calib_file_path)
        cam_width, cam_height = int(source.get(3)), int(source.get(4))
        new_cam_matrix, roi = cv.getOptimalNewCameraMatrix(cam_matrix, dist_coeff, (cam_width, cam_height), 1,
                                                      (cam_width, cam_height))

    # Setup video saving if enabled and initialize the video writer object
    if save_state:
        save_dir = "Recordings"
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        timestr = time.strftime("%Y%m%d_%H-%M-%S_")
        file_name = 'webcam_stream.mp4'
        save_path = os.path.join(save_dir, str(timestr + file_name))
        # Initialise videoWriterObject to store video
        video_writer = video_writer_object(source, save_path)

    # Main video loop
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

        # remove camera distortion if camera calibrated
        if 'new_cam_matrix' in locals():
            frame = cv.undistort(frame, cam_matrix, dist_coeff, None, new_cam_matrix)
            win_name = 'Camera Calibration'
            # Draw a green rectangle, illustrating the undistorted region
            x, y, w, h = roi
            color = (0, 255, 0)  # Green color in BGR format
            thickness = 2  # Thickness of the rectangle border
            cv.rectangle(frame, (x, y), (x + w, y + h), color, thickness)

        if save_state:
            # Write the frame to the output files
            video_writer.write(frame)

        global IMAGE_COUNTER
        global LAST_CAPTURE_TIME
        global DISPLAY_DURATION

        # Check if we need to display the captured image number
        current_time = int(time.time() * 1000)  # Get current time in milliseconds

        if current_time - LAST_CAPTURE_TIME <= DISPLAY_DURATION and IMAGE_COUNTER > 1 and not LAST_CAPTURE_TIME == 0:
            # Display the last captured image number for DISPLAY_DURATION time
            text = f"Captured calibration picture {IMAGE_COUNTER - 1}"
            (text_width, text_height), text_baseline = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            cv.putText(frame, text, (int(0.5*(frame_width-text_width)), int(0.95*frame_height-0.5*text_height)),
                       cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv.LINE_AA)

        # Display the resulting frame
        cv.imshow(win_name, frame)

        # Capture the key pressed
        key = cv.waitKey(1) & 0xFF
        # If spacebar is pressed, capture and save the image
        if key == 32:  # ASCII for space bar is 32
            img_name = f"calibration_{IMAGE_COUNTER}.jpg"
            img_path = os.path.join(save_dir_calib, img_name)
            cv.imwrite(img_path, frame)
            print(f"Image saved as {img_path}")
            IMAGE_COUNTER += 1
            LAST_CAPTURE_TIME = int(time.time() * 1000)  # Reset capture time
        elif key == 113 or key == 27 or cv.getWindowProperty(win_name, cv.WND_PROP_VISIBLE) < 1:
            break

    # Cleanup
    if save_state and video_writer:
        video_writer.release()
    source.release()
    cv.destroyAllWindows()


def run_pose_estimation(tag, save_state, tag_size):
    # Load your calibration data
    calib_file_path = "calibration.npz"
    rep_err, cam_matrix, dist_coeff, r_vec, t_vec = load_calibration_from_file(calib_file_path)

    # Select ArUco dictionary type
    aruco_dict_type = ARUCO_DICT[tag]

    # Initialize video capture
    video = cv.VideoCapture(0)
    win_name = 'Pose Estimation'

    # Initialize camera calibration if available
    calib_file_path = "calibration.npz"
    if os.path.exists(calib_file_path):
        rep_err, cam_matrix, dist_coeff, r_vec, t_vec = load_calibration_from_file(calib_file_path)
        cam_width, cam_height = int(video.get(3)), int(video.get(4))
        new_cam_matrix, roi = cv.getOptimalNewCameraMatrix(cam_matrix, dist_coeff, (cam_width, cam_height), 1,
                                                           (cam_width, cam_height))

    # Initialise storage of video recording
    if save_state:
        # Setup the save path and initialize the video writer
        save_dir = "Recordings"
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        timestr = time.strftime("%Y%m%d_%H-%M-%S_")
        file_name = 'webcam_stream.mp4'
        save_path = os.path.join(save_dir, str(timestr + file_name))
        # Initialise videoWriterObject to store video
        video_writer = video_writer_object(video, save_path)

    while True:
        ret, frame = video.read()
        if not ret:
            break

        # remove camera distortion if camera calibrated
        if 'new_cam_matrix' in locals():
            frame = cv.undistort(frame, cam_matrix, dist_coeff, None, new_cam_matrix)
            # crop image to undistorted region
            x, y, w, h = roi
            frame = frame[y:y + h, x:x + w]
            win_name = 'Undistorted Pose Estimation'

        # Call pose estimation function
        output = pose_estimation(frame, aruco_dict_type, new_cam_matrix, dist_coeff, tag_size)

        # Display result
        cv.imshow(win_name, output)

        if save_state:
            # Write the frame to the output files
            video_writer.write(frame)

        # Capture the key pressed
        key = cv.waitKey(1) & 0xFF
        # ESC or 'q' key or window closed
        if key in [27, 113] or cv.getWindowProperty(win_name, cv.WND_PROP_VISIBLE) < 1:
            break

    video.release()
    cv.destroyAllWindows()


# Test enhancement of aruco marker --> ArUcoE
def marker_enhancement(aruco_dict_tag):
    # initialise new DICT markers
    dict_enhanced = cv.aruco.Dictionary()

    # select predefined ArUco marker dictionary
    arucoDict = cv.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_dict_tag])
    # get bytesList of all ids
    bytes_list = arucoDict.bytesList
    max_corr_bits = arucoDict.maxCorrectionBits
    side_pixels = arucoDict.markerSize
    total_ids = bytes_list.shape[0]

    # assign size and max corr bites to new DICT
    dict_enhanced.markerSize = side_pixels + 4
    dict_enhanced.maxCorrectionBits = max_corr_bits

    # generate enhanced bits pattern for each of the markers available in the standard DICT and assign to enhanced DICT
    bytes_list_enhanced = np.empty((0,))
    for aruco_id in range(total_ids):
        # extract bits from selected ArUco tag
        next_aruco_id = aruco_id + 1
        bits_matrix = cv.aruco.Dictionary_getBitsFromByteList(bytes_list[aruco_id:next_aruco_id, :, :], side_pixels)

        # matrix enhancement
        vertical_vector_ones = np.ones((side_pixels, 1), dtype=np.uint8)
        vertical_vector_zeros = np.zeros((side_pixels, 1), dtype=np.uint8)
        # horizontal enhancement
        bits_horizontal_enhanced = np.hstack((vertical_vector_ones, vertical_vector_zeros,
                                              bits_matrix,
                                              vertical_vector_zeros, vertical_vector_ones))

        horizontal_vector_zeros = np.zeros((1, (side_pixels + 4)), dtype=np.uint8)
        # switch first & last bit to 1
        np.put(horizontal_vector_zeros, [0, -1], 1)
        # bit flip
        horizontal_vector_ones = 1 - horizontal_vector_zeros
        # vertical enhancement
        bits_enhanced = np.vstack([horizontal_vector_ones, horizontal_vector_zeros,
                                   bits_horizontal_enhanced,
                                   horizontal_vector_zeros, horizontal_vector_ones])

        # transform Bits to Bytes --> Storage type of Markers
        new_marker_comp = cv.aruco.Dictionary_getByteListFromBits(bits_enhanced)
        # Add the marker as a new row
        if bytes_list_enhanced.size == 0:
            # If the array is empty, directly reshape to match the first element
            bytes_list_enhanced = new_marker_comp
        else:
            # Otherwise, concatenate along the first axis
            bytes_list_enhanced = np.concatenate((bytes_list_enhanced, new_marker_comp), axis=0)

    # assign bytesList to enhanced DICT
    dict_enhanced.bytesList = bytes_list_enhanced
    # Generate the ArUco tag
    tag_size = 220
    tag_id = 1
    tag = np.zeros((tag_size, tag_size, 1), dtype="uint8")
    dict_enhanced.generateImageMarker(tag_id, tag_size, tag, 1)

    # Optionally, display the tag
    cv.imshow("ArUCoE Tag", tag)
    # Wait for key press or window close
    while True:
        key = cv.waitKey(1) & 0xFF
        if key in [27, 113]:  # ESC or 'q' key
            break
        if cv.getWindowProperty("ArUcoE Tag", cv.WND_PROP_VISIBLE) < 1:
            break

    cv.destroyAllWindows()
    print("done")


if __name__ == "__main__":
    cam_nr = 0  # Use internal camera
    cam_calibrate()  # Perform camera calibration
    save_video_state, pose_estimation_state, aruco_tag, aruco_size = user_requests()
    # Generate ArUco tag with user-specified parameters
    id_tag = list(ARUCO_DICT.keys()).index(aruco_tag)+1
    # marker_enhancement(aruco_tag)
    generate_aruco_tag(output_path="tags", tag_id=id_tag, tag_type=aruco_tag,
                       tag_size=aruco_size)
    if pose_estimation_state:
        run_pose_estimation(aruco_tag, save_video_state, aruco_size)
    else:
        stream_video(cam_nr, save_video_state)


