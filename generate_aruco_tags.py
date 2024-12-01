import argparse
import os
import numpy as np
import cv2
from calibrate import ARUCO_DICT


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

'''
# Test enhancement of aruco marker --> ArUcoE
def marker_enhancement(aruco_dict_tag):
    # initialise new DICT markers
    dict_enhanced = cv2.aruco.Dictionary()

    # select predefined ArUco marker dictionary
    arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_dict_tag])
    # get bytesList of all ids
    bytes_list = arucoDict.bytesList
    max_corr_bits = arucoDict.maxCorrectionBits
    side_pixels = arucoDict.markerSize
    total_ids = bytes_list.shape[0]

    # assign size and max corr bites to new DICT
    side_pixels_enhanced = side_pixels + 6
    dict_enhanced.markerSize = side_pixels_enhanced
    dict_enhanced.maxCorrectionBits = max_corr_bits

    # generate enhanced bits pattern for each of the markers available in the standard DICT and assign to enhanced DICT
    bytes_list_enhanced = np.empty((0,))
    for aruco_id in range(total_ids):
        # extract bits from selected ArUco tag
        next_aruco_id = aruco_id + 1
        bits_matrix = cv2.aruco.Dictionary_getBitsFromByteList(bytes_list[aruco_id:next_aruco_id, :, :], side_pixels)

        # matrix enhancement
        vertical_vector_ones = np.ones((side_pixels, 1), dtype=np.uint8)
        vertical_vector_zeros = np.zeros((side_pixels, 1), dtype=np.uint8)
        # horizontal enhancement
        bits_horizontal_enhanced = np.hstack((vertical_vector_zeros, vertical_vector_ones, vertical_vector_zeros,
                                              bits_matrix,
                                              vertical_vector_zeros, vertical_vector_ones, vertical_vector_zeros))

        horizontal_vector_zeros = np.zeros((1, side_pixels_enhanced), dtype=np.uint8)
        # switch first & last bit to 1
        np.put(horizontal_vector_zeros, [0, -1], 1)
        # switch place of first item with second one and second last with last one
        horizontal_vector_zeros_inter = np.zeros((1, side_pixels_enhanced), dtype=np.uint8)
        # switch second & second last bit to 1
        np.put(horizontal_vector_zeros_inter, [1, -2], 1)

        # bit flip
        horizontal_vector_ones = 1 - horizontal_vector_zeros
        # vertical enhancement
        bits_enhanced = np.vstack([horizontal_vector_zeros, horizontal_vector_ones, horizontal_vector_zeros_inter,
                                   bits_horizontal_enhanced,
                                   horizontal_vector_zeros_inter, horizontal_vector_ones, horizontal_vector_zeros])

        # transform Bits to Bytes --> Storage type of Markers
        new_marker_comp = cv2.aruco.Dictionary_getByteListFromBits(bits_enhanced)
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
    # Todo: currently fixed sizing factor --> adjust so it is as in standard version
    # Todo: print values in translational and rotational values
    tag_size = side_pixels_enhanced * 10
    tag_id = 1
    tag = np.zeros((tag_size, tag_size, 1), dtype="uint8")
    dict_enhanced.generateImageMarker(tag_id, tag_size, tag, 1)

    # Optionally, display the tag
    cv2.imshow("ArUCoE Tag", tag)
    # Wait for key press or window close
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key in [27, 113]:  # ESC or 'q' key
            break
        if cv2.getWindowProperty("ArUcoE Tag", cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()
    print("done")
    '''


# Support both direct execution and import as a module
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", required=True, help="Path to output folder to save ArUCo tag")
    ap.add_argument("-i", "--id", type=int, required=False, help="ID of ArUCo tag to generate")
    ap.add_argument("-t", "--type", type=str, default="DICT_ARUCO_ORIGINAL", help="Type of ArUCo tag to generate")
    ap.add_argument("-s", "--size", type=int, default=200, help="Size of the ArUCo tag")
    args = vars(ap.parse_args())
    if not args["id"]:
        args["id"] = list(ARUCO_DICT.keys()).index(args["type"]) + 1
    # Call the function with parsed arguments
    generate_aruco_tag(args["output"], args["id"], args["type"], args["size"])
