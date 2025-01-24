import os
import cv2
import numpy as np
import shutil
import pyrealsense2 as rs

# Create directories to save color and depth frames
obj_dir = "cup_test"
color_dir = 'rgb'
depth_dir = 'depth'

obj_color_dir = os.path.join(obj_dir, color_dir)
obj_depth_dir = os.path.join(obj_dir, depth_dir)
if os.path.exists(obj_color_dir):
    shutil.rmtree(obj_color_dir)
os.makedirs(obj_color_dir)

if os.path.exists(obj_depth_dir):
    shutil.rmtree(obj_depth_dir)
os.makedirs(obj_depth_dir)

# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
# different resolutions of color and depth streams
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 6)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 6)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: ", depth_scale)

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

# Streaming loop
try:
    frame_count = 0
    while True:
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Scale depth image to mm
        depth_image_scaled = (depth_image * depth_scale * 1000).astype(np.uint16)

        # Save the frames
        for ending in ["png", "jpg"]:
            color_filename = os.path.join(obj_color_dir, f'{frame_count:07d}.{ending}')
            cv2.imwrite(color_filename, color_image)

        # Save depth image in millimeters (16-bit PNG format)
        depth_filename = os.path.join(obj_depth_dir, f'{frame_count:07d}.png')
        cv2.imwrite(depth_filename, depth_image_scaled)

        print(f'Saved {color_filename} and {depth_filename}')
        frame_count += 1

        # Display the frames
        cv2.imshow('Color Frame', color_image)

        # Press 'q' on the keyboard to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()
