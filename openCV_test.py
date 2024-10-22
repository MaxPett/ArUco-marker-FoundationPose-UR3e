import cv2 as cv
import time
import tkinter as tk
import os


def stream_video(cam_id, save_state):
    source = cv.VideoCapture(cam_id)
    win_name = 'Webcam Feed'

    # Check if the webcam is opened correctly
    if not source.isOpened():
        print("Error: Could not open webcam.")
        exit()

    while True:
        # Capture frame-by-frame from the webcam
        has_frame, frame = source.read()
        # If frame was not captured correctly, exit the loop
        if not has_frame:
            print("Error: Failed to capture image.")
            break

        # flip source
        frame = cv.flip(frame, 1)

        # Display the resulting frame
        cv.imshow(win_name, frame)

        if save_state:
            save_dir = "Recordings"
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)

            timestr = time.strftime("%Y%m%d-%H:%M:%S_")
            file_name = 'webcam_stream.mp4'
            save_path = os.path.join(save_dir, "/", timestr, file_name)
            # Initialise videoWriter to store video
            video_writer = video_writer_object(source, save_path)
            # Write the frame to the output files
            video_writer.write(frame)

        # Exit the loop if any key is pressed or the close button is pressed
        if cv.waitKey(1) != -1 or cv.getWindowProperty(win_name, cv.WND_PROP_VISIBLE) < 1:
            break

    # Release the video source and close any OpenCV windows
    source.release()
    video_writer.release()
    cv.destroyAllWindows()


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
        out_mp4 = cv.VideoWriter(video_name, cv.VideoWriter_fourcc(*"XVID"), 10, (frame_width, frame_height))
        return out_mp4


def save_video(cam_id):
    cap = cv.VideoCapture(cam_id)

    if not cap.isOpened():
        print("Error opening video stream or file")

    # Read until video is completed
    while cap.isOpened():
        # Capture frame-by-frame
        has_frame, frame = cap.read()

        if has_frame:
            # Write the frame to the output files
            video_writer = video_writer_object(cap, 'webcam_stream.mp4')
            video_writer.write(frame)

        # Break the loop
        else:
            break

    # When everything done, release the VideoCapture and VideoWriter objects
    cap.release()
    video_writer.release()


if __name__ == "__main__":
    cam_nr = 0
    save_video_state = video_save_request()
    stream_video(cam_nr, save_video_state)






