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

############################################################################





if __name__ == "__main__":
    os.system(f"python gen_pattern.py -o {CALIB_TYPE}.svg -r {CALIB_ROWS} -c {CALIB_COLUMNS} -T {CALIB_TYPE} -s {CALIB_PATTERN_SIZE} -a {CALIB_PAGE_SIZE}")

    print('DONE!')


