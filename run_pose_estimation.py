import cv2 as cv
import glob
import time
import tkinter as tk
from tkinter import ttk, messagebox
import os
import subprocess
import numpy as np


CAM_NR = 0   # 0 use internal camera
CALIB_ROWS = 11
CALIB_COLUMNS = 8
CALIB_TYPE = 'checkerboard'  # circles, acircles, checkerboard, radon_checkerboard, charuco_board.
CALIB_PATTERN_SIZE = 20.0
CALIB_PAGE_SIZE = 'A4'
CALIB_IMG_PATH = 'calibration'

############################################################################


def check_execution(message):
    # Check the result
    if message.returncode == 0:
        print("Command executed successfully")
        print("Output:", message.stdout)
    else:
        print("Command failed with return code", message.returncode)
        print("Error:", message.stderr)


def execute_pattern_generation(columns, rows, calib_type, pattern_name, pattern_size, page_size):
    # Construct the command with mandatory elements
    command = ["python", "gen_pattern.py",
               "-o", str(pattern_name) + ".svg",
               "-r", str(rows),
               "-c", str(columns),
               "-T", calib_type,
               "-s", str(pattern_size),
               "-a", page_size]
    # more options available!
    result = subprocess.run(command, capture_output=True, text=True)
    check_execution(result)


def execute_calibration(columns, rows, calib_type, camera_id, path, pattern_size):
    # Construct the command with mandatory elements
    command = ["python", "calibrate.py",
               "-c", str(columns),
               "-r", str(rows),
               "-t", calib_type,
               "-n", str(camera_id),
               "-p", path,
               "--square_size", str(pattern_size)]
    # more options available!

    result = subprocess.run(command, capture_output=True, text=True)
    check_execution(result)


if __name__ == "__main__":
    CALIB_COLUMNS = 7
    CALIB_ROWS = 9
    execute_pattern_generation(CALIB_COLUMNS, CALIB_ROWS, CALIB_TYPE, CALIB_TYPE, CALIB_PATTERN_SIZE, CALIB_PAGE_SIZE)
    # check if calib images of correct pattern
    execute_calibration(CALIB_COLUMNS, CALIB_ROWS, CALIB_TYPE, CAM_NR, CALIB_IMG_PATH, CALIB_PATTERN_SIZE)
    print('DONE!')


