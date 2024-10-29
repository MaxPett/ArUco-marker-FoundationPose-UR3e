# gui.py
"""GUI components for the camera calibration system."""

import tkinter as tk
from tkinter import ttk, messagebox
from dataclasses import dataclass
from typing import Optional
import cv2 as cv

# This would typically be imported from another module
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


@dataclass
class UserPreferences:
    """Dataclass to store user preferences for video settings."""
    save_video: bool
    run_pose_estimation: bool
    aruco_type: str
    video_size: int


class SettingsDialog:
    """Dialog window for collecting user preferences for video settings."""

    def __init__(self, parent=None):
        """Initialize the settings dialog.

        Args:
            parent: Optional parent window for the dialog
        """
        self.window = tk.Tk() if parent is None else tk.Toplevel(parent)
        self.window.title("Video Settings")
        self.window.geometry("500x500+650+250")

        # GUI control variables
        self.save_yes_var = tk.BooleanVar()
        self.save_no_var = tk.BooleanVar()
        self.pose_yes_var = tk.BooleanVar()
        self.pose_no_var = tk.BooleanVar()
        self.aruco_type = tk.StringVar()
        self.video_size = tk.StringVar(value="200")

        self.result = None
        self._setup_ui()

    def _validate_size(self, value: str) -> bool:
        """Validate the ArUco marker size input.

        Args:
            value: The string value to validate

        Returns:
            bool: True if the value is valid, False otherwise
        """
        if value == "":
            return True  # Allow empty field for typing
        try:
            size = int(value)
            return size > 0  # Only allow positive integers
        except ValueError:
            return False

    def _on_save_yes(self):
        """Handle save video Yes checkbox."""
        if self.save_yes_var.get():
            self.save_no_var.set(False)

    def _on_save_no(self):
        """Handle save video No checkbox."""
        if self.save_no_var.get():
            self.save_yes_var.set(False)

    def _on_pose_yes(self):
        """Handle pose estimation Yes checkbox."""
        if self.pose_yes_var.get():
            self.pose_no_var.set(False)

    def _on_pose_no(self):
        """Handle pose estimation No checkbox."""
        if self.pose_no_var.get():
            self.pose_yes_var.set(False)

    def _on_submit(self):
        """Handle form submission and validation."""
        # Validate all inputs
        if not (self.save_yes_var.get() or self.save_no_var.get()):
            messagebox.showerror("Error", "Please select Yes or No for saving video")
            return
        if not (self.pose_yes_var.get() or self.pose_no_var.get()):
            messagebox.showerror("Error", "Please select Yes or No for pose estimation")
            return

        try:
            size = int(self.video_size.get())
            if size <= 0:
                messagebox.showerror("Error", "Please enter a positive number for size")
                return
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number for size")
            return

        # Create UserPreferences object with the collected data
        self.result = UserPreferences(
            save_video=self.save_yes_var.get(),
            run_pose_estimation=self.pose_yes_var.get(),
            aruco_type=self.aruco_type.get(),
            video_size=int(self.video_size.get())
        )

        self.window.destroy()

    def _setup_ui(self):
        """Set up the user interface elements."""
        validate_cmd = self.window.register(self._validate_size)

        main_frame = ttk.Frame(self.window, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Save video question
        save_label = ttk.Label(main_frame, text="Do you want to save the following video?", font=("Arial", 12))
        save_label.pack(pady=10)

        save_frame = ttk.Frame(main_frame)
        save_frame.pack(pady=5)
        save_yes_check = ttk.Checkbutton(save_frame, text="Yes", variable=self.save_yes_var, command=self._on_save_yes)
        save_yes_check.pack(side=tk.LEFT, padx=10)
        save_no_check = ttk.Checkbutton(save_frame, text="No", variable=self.save_no_var, command=self._on_save_no)
        save_no_check.pack(side=tk.LEFT, padx=10)

        # Pose estimation question
        pose_label = ttk.Label(main_frame, text="Do you want to run pose estimation?", font=("Arial", 12))
        pose_label.pack(pady=10)

        pose_frame = ttk.Frame(main_frame)
        pose_frame.pack(pady=5)
        pose_yes_check = ttk.Checkbutton(pose_frame, text="Yes", variable=self.pose_yes_var, command=self._on_pose_yes)
        pose_yes_check.pack(side=tk.LEFT, padx=10)
        pose_no_check = ttk.Checkbutton(pose_frame, text="No", variable=self.pose_no_var, command=self._on_pose_no)
        pose_no_check.pack(side=tk.LEFT, padx=10)

        # ArUco dictionary dropdown
        aruco_label = ttk.Label(main_frame, text="Select ArUco Dictionary:", font=("Arial", 12))
        aruco_label.pack(pady=10)
        aruco_dropdown = ttk.Combobox(main_frame, textvariable=self.aruco_type, values=list(ARUCO_DICT.keys()),
                                      state="readonly")
        aruco_dropdown.pack()
        aruco_dropdown.set(list(ARUCO_DICT.keys())[0])  # Set default value

        # ArUco size input
        size_frame = ttk.Frame(main_frame)
        size_frame.pack(pady=10)

        size_label = ttk.Label(size_frame, text="Enter ArUco size (pixels):", font=("Arial", 12))
        size_label.pack(side=tk.LEFT, padx=5)

        size_entry = ttk.Entry(size_frame, textvariable=self.video_size, width=10,
                               validate="key", validatecommand=(validate_cmd, '%P'))
        size_entry.pack(side=tk.LEFT, padx=5)

        # Submit button
        submit_button = ttk.Button(main_frame, text="Submit", command=self._on_submit)
        submit_button.pack(pady=20)

    def get_settings(self) -> Optional[UserPreferences]:
        """Show the dialog and return the user preferences.

        Returns:
            Optional[UserPreferences]: The user's preferences, or None if canceled
        """
        self.window.mainloop()
        return self.result