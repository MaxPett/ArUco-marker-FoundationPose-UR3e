import os
from PIL import Image

def convert_png_to_jpg(directory_path):
    """
    Converts all .png images in the given directory to .jpg format.

    Args:
        directory_path (str): Path to the directory containing .png images.
    """
    if not os.path.isdir(directory_path):
        print(f"Error: {directory_path} is not a valid directory.")
        return

    for filename in os.listdir(directory_path):
        if filename.lower().endswith('.png'):
            png_path = os.path.join(directory_path, filename)
            jpg_path = os.path.join(directory_path, os.path.splitext(filename)[0] + '.jpg')

            try:
                with Image.open(png_path) as img:
                    rgb_img = img.convert('RGB')
                    rgb_img.save(jpg_path, 'JPEG')
                    print(f"Converted {filename} to {os.path.basename(jpg_path)}")
            except Exception as e:
                print(f"Failed to convert {filename}: {e}")

if __name__ == "__main__":
    directory = input("Enter the directory path: ")
    convert_png_to_jpg(directory)
