import tkinter as tk
from tkinter import ttk, messagebox
from calibrate import ARUCO_DICT
import re  # For IP address validation


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
    new_window.geometry("500x550+650+250")

    # GUI control variables
    save_result = tk.BooleanVar()
    aruco_type = tk.StringVar()
    video_size = tk.StringVar(value="200")
    test_object_name = tk.StringVar(value="ArUco")
    robot_ip = tk.StringVar(value="192.168.1.3")
    save_yes_var = tk.BooleanVar()
    save_no_var = tk.BooleanVar()
    pose_type = tk.StringVar(value="Foundation Pose")

    # Input validation for ArUco marker size
    def validate_size(value):
        if value == "": return True  # Allow empty field for typing
        try:
            size = int(value)
            return size > 0  # Only allow positive integers
        except ValueError:
            return False

    # Input validation for IP address
    def validate_ip(ip):
        pattern = re.compile(
            r"^((25[0-5]|2[0-4][0-9]|[0-1]?[0-9][0-9]?)\.){3}"
            r"(25[0-5]|2[0-4][0-9]|[0-1]?[0-9][0-9]?)$"
        )
        return pattern.match(ip) is not None

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

        if pose_type.get() == "ArUco Pose":
            # Validate ArUco size
            try:
                size = int(video_size.get())
                if size <= 0:
                    messagebox.showerror("Error", "Please enter a positive number for size")
                    return
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid number for size")
                return

        # Validate IP address if saving video
        if save_yes_var.get() and not validate_ip(robot_ip.get()):
            messagebox.showerror("Error", "Please enter a valid IP address")
            return

        save_result.set(save_yes_var.get())  # True if Yes is checked, False if No is checked
        new_window.destroy()

    # GUI Layout setup
    main_frame = ttk.Frame(new_window, padding="20")
    main_frame.pack(fill=tk.BOTH, expand=True)

    # Test object name input
    test_object_label = ttk.Label(main_frame, text="Enter Test Object Name:", font=("Arial", 12))
    test_object_label.pack(pady=10)
    test_object_entry = ttk.Entry(main_frame, textvariable=test_object_name)
    test_object_entry.pack()

    # Save video question
    save_label = ttk.Label(main_frame, text="Do you want to save the following video?", font=("Arial", 12))
    save_label.pack(pady=10)

    save_frame = ttk.Frame(main_frame)
    save_frame.pack(pady=5)
    save_yes_check = ttk.Checkbutton(save_frame, text="Yes", variable=save_yes_var, command=on_save_yes)
    save_yes_check.pack(side=tk.LEFT, padx=10)
    save_no_check = ttk.Checkbutton(save_frame, text="No", variable=save_no_var, command=on_save_no)
    save_no_check.pack(side=tk.LEFT, padx=10)

    # Robot IP address input (only shown if video is saved)
    ip_label = ttk.Label(main_frame, text="Enter Robot IP Address:", font=("Arial", 12))
    ip_label.pack(pady=10)
    ip_entry = ttk.Entry(main_frame, textvariable=robot_ip)
    ip_entry.pack()

    # Pose type dropdown
    pose_label = ttk.Label(main_frame, text="Select Pose Type:", font=("Arial", 12))
    pose_label.pack(pady=10)
    pose_dropdown = ttk.Combobox(main_frame, textvariable=pose_type, values=["ArUco Pose", "Foundation Pose"], state="readonly")
    pose_dropdown.pack()
    pose_dropdown.set("Foundation Pose")  # Set default value

    # ArUco dictionary dropdown and size input (only shown if ArUco Pose is selected)
    def update_aruco_options(*args):
        if pose_type.get() == "ArUco Pose":
            aruco_label.pack(pady=10)
            aruco_dropdown.pack()
            size_frame.pack(pady=10)
            submit_button.pack_forget()  # Temporarily remove submit button
            submit_button.pack(pady=20)  # Re-add submit button at the end
        else:
            aruco_label.pack_forget()
            aruco_dropdown.pack_forget()
            size_frame.pack_forget()
            submit_button.pack_forget()  # Temporarily remove submit button
            submit_button.pack(pady=20)  # Re-add submit button at the end

    pose_type.trace_add("write", update_aruco_options)

    aruco_label = ttk.Label(main_frame, text="Select ArUco Dictionary:", font=("Arial", 12))
    aruco_dropdown = ttk.Combobox(main_frame, textvariable=aruco_type, values=list(ARUCO_DICT.keys()), state="readonly")
    aruco_dropdown.set(list(ARUCO_DICT.keys())[12])  # Set default value

    size_frame = ttk.Frame(main_frame)
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

    # Return the results as a tuple (save_video, aruco_type, video_size, pose_type)
    return save_result.get(), aruco_type.get(), int(video_size.get()) if pose_type.get() == "ArUco Pose" else None, test_object_name.get(), robot_ip.get(), pose_type.get()


if __name__ == "__main__":
    user_requests()
