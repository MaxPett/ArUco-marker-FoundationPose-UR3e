import os
import shutil

def split_and_copy_images(input_folder, output_folder_prefix, split_size):
    # Define input subfolders
    depth_folder = os.path.join(input_folder, "depth")
    rgb_folder = os.path.join(input_folder, "rgb")

    # Validate input folders
    if not os.path.exists(depth_folder) or not os.path.exists(rgb_folder):
        raise FileNotFoundError("Both 'depth' and 'rgb' subfolders must exist in the input folder.")

    # Get sorted list of RGB images (timestamps are used for matching, hence sorted order matters)
    rgb_images = sorted([f for f in os.listdir(rgb_folder) if f.endswith(".jpg") or f.endswith(".png")])
    depth_images = sorted([f for f in os.listdir(depth_folder) if f.endswith(".png")])

    # Split RGB images into chunks of `split_size` based on .jpg images only
    jpg_images = [f for f in rgb_images if f.endswith(".jpg")]
    for idx, start in enumerate(range(0, len(jpg_images), split_size)):
        end = start + split_size
        chunk = jpg_images[start:end]

        # Define the new folder structure
        new_folder = f"{output_folder_prefix}_{idx}"
        new_rgb_folder = os.path.join(new_folder, "rgb")
        new_depth_folder = os.path.join(new_folder, "depth")

        # Create the new subfolders
        os.makedirs(new_rgb_folder, exist_ok=True)
        os.makedirs(new_depth_folder, exist_ok=True)

        # Copy RGB images and their corresponding depth images
        for jpg_image in chunk:
            # Copy JPG image
            jpg_src_path = os.path.join(rgb_folder, jpg_image)
            jpg_dst_path = os.path.join(new_rgb_folder, jpg_image)
            shutil.copy(jpg_src_path, jpg_dst_path)

            # Copy corresponding PNG in RGB folder if exists
            rgb_png_image = jpg_image.replace(".jpg", ".png")
            rgb_png_src_path = os.path.join(rgb_folder, rgb_png_image)
            if os.path.exists(rgb_png_src_path):
                rgb_png_dst_path = os.path.join(new_rgb_folder, rgb_png_image)
                shutil.copy(rgb_png_src_path, rgb_png_dst_path)

            # Derive corresponding depth image name and copy it
            depth_image = jpg_image.replace(".jpg", ".png")
            depth_src_path = os.path.join(depth_folder, depth_image)
            if os.path.exists(depth_src_path):
                depth_dst_path = os.path.join(new_depth_folder, depth_image)
                shutil.copy(depth_src_path, depth_dst_path)

if __name__ == "__main__":
    # Input folder containing 'depth' and 'rgb' subfolders
    input_folder = "../out"
    # Output folder prefix (e.g., 'folder')
    output_folder_prefix = "out_splited"
    # Number of images per split
    split_size = 1500

    split_and_copy_images(input_folder, output_folder_prefix, split_size)
