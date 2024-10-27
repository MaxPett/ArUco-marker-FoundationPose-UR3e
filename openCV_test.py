import cv2 as cv
import glob
import time
import tkinter as tk
import os
import numpy as np
from generate_aruco_tags import generate_aruco_tag


# Global variable to track image count and display time for the last captured image
IMAGE_COUNTER = 1
LAST_CAPTURE_TIME = 0
DISPLAY_DURATION = 500  # Display the image number for 1000ms (1 second)


def cam_calibrate(showPics=True):
    # @ Kevin Wood
    # Read Image
    root = os.getcwd()
    calibration_dir = os.path.join(root, "calibration")
    list_img_paths = glob.glob(os.path.join(calibration_dir, "*.png"))

    # Start capturing calibration pictures
    if len(list_img_paths) == 0:
        stream_video(0, False)
        list_img_paths = glob.glob(os.path.join(calibration_dir, "*.png"))

    # Initialize
    n_rows = 8
    m_cols = 6
    term_crit = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    world_points_cur = np.zeros((n_rows*m_cols, 3), np.float32)
    world_points_cur[:, :2] = np.mgrid[0:n_rows, 0:m_cols].T.reshape(-1, 2)
    list_world_points = []
    list_img_points = []

    # Find Corners
    for cur_img_path in list_img_paths:
        img_BGR = cv.imread(cur_img_path)
        img_gray = cv.cvtColor(img_BGR, cv.COLOR_BGR2GRAY)
        cor_found, cor_org = cv.findChessboardCorners(img_gray, (n_rows, m_cols), None)

        if cor_found:
            list_world_points.append(world_points_cur)
            cor_refined = cv.cornerSubPix(img_gray, cor_org, (11, 11), (-1, -1), term_crit)
            list_img_points.append(cor_refined)
            if showPics:
                cv.drawChessboardCorners(img_BGR, (n_rows, m_cols), cor_refined, cor_found)
                cv.imshow('Chessboard', img_BGR)
                cv.waitKey(1000)
    cv.destroyAllWindows()

    # Calibrate
    rep_err, cam_matrix, dist_coeff, r_vec, t_vec = cv.calibrateCamera(list_world_points, list_img_points,
                                                                       img_gray.shape[::-1], None, None)
    print('Camera Matrix:\n', cam_matrix)
    print("Reproj Error (pixles): {:.4f}".format(rep_err))

    curr_folder = os.path.dirname(os.path.abspath(__file__))
    param_path = os.path.join(curr_folder, 'calibration.npz')
    np.savez(param_path,
             rep_err=rep_err,
             cam_matrix=cam_matrix,
             dist_coeff=dist_coeff,
             r_vec=r_vec,
             t_vec=t_vec)

    return cam_matrix, dist_coeff


def video_save_request():
    # Create a new window
    new_window = tk.Tk()
    new_window.title("Save Video Request")

    # Set the size of the window (decent size)
    new_window.geometry("300x150+650+250")

    # Create a variable to hold the result
    result = tk.BooleanVar()

    # Function to handle 'Yes' button click
    def on_yes():
        result.set(True)
        new_window.destroy()  # Close the window

    # Function to handle 'No' button click
    def on_no():
        result.set(False)
        new_window.destroy()  # Close the window

    # Create a label with a question
    label = tk.Label(new_window, text="Do you want to save the following video?", font=("Arial", 12))
    label.pack(pady=20)

    # Create 'Yes' and 'No' buttons
    yes_button = tk.Button(new_window, text="Yes", width=10, command=on_yes)
    yes_button.pack(side="left", padx=20, pady=10)

    no_button = tk.Button(new_window, text="No", width=10, command=on_no)
    no_button.pack(side="right", padx=20, pady=10)

    # Wait for the window to close before returning the result
    new_window.mainloop()

    # Return the result (True or False)
    return result.get()


def video_writer_object(source, video_name):
    # Default resolutions of the frame are obtained, fourcc code and frame dimensions important!
    # Convert the resolutions from float to integer.
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


def stream_video(cam_id, save_state):
    source = cv.VideoCapture(cam_id)
    win_name = 'Webcam Feed'

    # Check if the webcam is opened correctly
    if not source.isOpened():
        print("Error: Could not open webcam.")
        exit()

    # Setup the save directory for captured images
    save_dir_calib = "calibration"
    if not os.path.exists(save_dir_calib):
        os.mkdir(save_dir_calib)

    calib_file_path = "calibration.npz"
    if os.path.exists(calib_file_path):
        # Load the camera calibration file
        with np.load(calib_file_path) as data:
            # Access each parameter
            rep_err = data['rep_err']
            cam_matrix = data['cam_matrix']
            dist_coeff = data['dist_coeff']
            r_vec = data['r_vec']
            t_vec = data['t_vec']
        cam_width, cam_height = int(source.get(3)), int(source.get(4))
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
        video_writer = video_writer_object(source, save_path)

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
        if new_cam_matrix.any():
            frame = cv.undistort(frame, cam_matrix, dist_coeff, None, new_cam_matrix)

        if save_state:
            # Write the frame to the output files
            video_writer.write(frame)

        global IMAGE_COUNTER
        global LAST_CAPTURE_TIME
        global DISPLAY_DURATION

        # Check if we need to display the captured image number
        current_time = int(time.time() * 1000)  # Get current time in milliseconds

        if current_time - LAST_CAPTURE_TIME <= DISPLAY_DURATION and IMAGE_COUNTER > 1 and not LAST_CAPTURE_TIME == 0:
            # Display the last captured image number for 1000ms
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
            img_name = f"calibration_{IMAGE_COUNTER}.png"
            img_path = os.path.join(save_dir_calib, img_name)
            cv.imwrite(img_path, frame)
            print(f"Image saved as {img_path}")
            IMAGE_COUNTER += 1
            LAST_CAPTURE_TIME = int(time.time() * 1000)  # Reset capture time

        # Exit the loop if any key is pressed or the close button is pressed
        if key == 113 or key == 27 or cv.getWindowProperty(win_name, cv.WND_PROP_VISIBLE) < 1:
            break

    # Release the video source and close any OpenCV windows
    if save_state and video_writer:
        video_writer.release()
    source.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    cam_nr = 0
    generate_aruco_tag(output_path="tags", tag_id=24, tag_type="DICT_5X5_100", tag_size=200)
    cam_calibrate()
    save_video_state = video_save_request()
    stream_video(cam_nr, save_video_state)







