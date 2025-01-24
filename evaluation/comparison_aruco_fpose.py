import os
import numpy as np
import pandas as pd
import glob
import cv2 as cv
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(aruco_path, fpose_path):
    df_aruco_data = pd.read_csv(aruco_path)
    df_fpose_data = pd.read_csv(fpose_path)
    return df_aruco_data, df_fpose_data


if __name__ == '__main__':
    aruco_data = "data_aruco_pose.csv"
    fpose_data = "data_foundation_pose.csv"
    df_aruco, df_fpose = load_data(aruco_data, fpose_data)
