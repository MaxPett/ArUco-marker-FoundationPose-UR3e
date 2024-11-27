import argparse
import os
import numpy as np
import cv2
from utils import ARUCO_DICT


# Adapted form GSNCodes ArUCo-Markers-Pose-Estimation-Generation-Python
# https://github.com/GSNCodes/ArUCo-Markers-Pose-Estimation-Generation-Python

def generate_aruco_tag(output_path, tag_id, tag_type="DICT_ARUCO_ORIGINAL", tag_size=200):
    """
    Generates an ArUCo tag and saves it to the specified output path.

    Parameters:
    - output_path (str): Path to save the generated ArUCo tag image.
    - tag_id (int): ID of the ArUCo tag to generate.
    - tag_type (str): Type of ArUCo tag to generate.
    - tag_size (int): Size of the ArUCo tag in pixels (optional, default=200).
    """
    # Check if the dictionary type is supported
    if ARUCO_DICT.get(tag_type, None) is None:
        print(f"ArUCo tag type '{tag_type}' is not supported")
        return

    # Get the dictionary and generate the tag
    arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[tag_type])
    print(f"Generating ArUCo tag of type '{tag_type}' with ID '{tag_id}'")

    # Generate the ArUco tag
    tag = np.zeros((tag_size, tag_size, 1), dtype="uint8")
    arucoDict.generateImageMarker(tag_id, tag_size, tag, 1)

    # Save the generated tag
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    tag_name = f'{output_path}/{tag_type}_id_{tag_id}.jpg'
    cv2.imwrite(tag_name, tag)
    print(f"ArUCo tag saved to {tag_name}")

    # Optionally, display the tag
    cv2.imshow("ArUCo Tag", tag)

    # Wait for key press or window close
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key in [27, 113]:  # ESC or 'q' key
            break
        if cv2.getWindowProperty("ArUCo Tag", cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()


# Support both direct execution and import as a module
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", required=True, help="Path to output folder to save ArUCo tag")
    ap.add_argument("-i", "--id", type=int, required=True, help="ID of ArUCo tag to generate")
    ap.add_argument("-t", "--type", type=str, default="DICT_ARUCO_ORIGINAL", help="Type of ArUCo tag to generate")
    ap.add_argument("-s", "--size", type=int, default=200, help="Size of the ArUCo tag")
    args = vars(ap.parse_args())

    # Call the function with parsed arguments
    generate_aruco_tag(args["output"], args["id"], args["type"], args["size"])