import matplotlib.pyplot as plt
# from imageio import imread
import numpy as np
import cv2
from glob import glob

image_paths = glob('frames/frame_*.jpg')
# label_paths = [p.replace('leftImg8bit', 'gtFine').replace(
#     '_image.jpg', '_label.png') for p in image_paths]

# Assigning some RGB colors for the 7 + 1 (Misc) classes

CITYSCAPES_COLORMAP = np.array([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0],  [0, 192, 0], [128, 192, 0],
                [0, 64, 128]], dtype=np.int64)

print(len(CITYSCAPES_COLORMAP))
# colors = np.array([
#     [128, 64, 18],      # Drivable
#     [244, 35, 232],     # Non Drivable
#     [220, 20, 60],      # Living Things
#     [0, 0, 230],        # Vehicles
#     [220, 190, 40],     # Road Side Objects
#     [70, 70, 70],       # Far Objects
#     [70, 130, 180],     # Sky
#     [0, 0, 0]           # Misc
# ], dtype=np.int)

print(len(image_paths))

video_path = "Road_1101.mp4"
#Function to extract frames from a video
def FrameCapture(video_path):
    
    vidObj = cv2.VideoCapture(video_path)
    count = 0
    success = 1
    
    while success:
        success, image = vidObj.read()
        # print(success, image)
        
        cv2.imwrite(f"frames/frame_{count}.jpg", image)
        count += 1

FrameCapture(video_path)

# for i in range(len(image_paths)):
#     print(image_paths[i], label_paths[i])
#     image_frame = imread(image_paths[i])
#     # print(image_frame.shape)
#     label_map = imread(label_paths[i])
#     color_image = np.zeros(
#         (label_map.shape[0], label_map.shape[1], 3), dtype=np.int)
#     for i in range(7):
#         color_image[label_map == i] = colors[i]

#     color_image[label_map == 255] = colors[7]
#     plt.imshow(image_frame)
#     plt.imshow(color_image, alpha=0.8)
#     plt.show()
