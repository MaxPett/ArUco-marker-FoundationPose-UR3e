import tkinter as tk
from tkinter import ttk, messagebox
from calibrate import ARUCO_DICT


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
    aruco_type = tk.StringVar()
    video_size = tk.StringVar(value="200")
    save_yes_var = tk.BooleanVar()
    save_no_var = tk.BooleanVar()

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

    # Form submission handler
    def on_submit():
        # Validate all inputs before proceeding
        if not (save_yes_var.get() or save_no_var.get()):
            messagebox.showerror("Error", "Please select Yes or No for saving video")
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

    # Return the results as a tuple (save_video, aruco_type, video_size)
    return save_result.get(), aruco_type.get(), int(video_size.get())


if __name__ == "__main__":
    user_requests()
