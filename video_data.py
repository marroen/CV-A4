import cv2 as cv
import torch
import time
import matplotlib.pyplot as plt

from torchvision import transforms
from PIL import Image

video_path = "video.mp4"
vid = cv.VideoCapture(video_path)

# Define transform to fit images with model
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((112, 112)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

processed_frames = []
frame_count = 0
grabbed_frames = 0

# Iterate through each frame
while True:

    success, frame = vid.read()  # Reads the next frame
    if not success:
        break  # Break when video ends

    # Append one frame every 80 frames, for a total of 50 frames across the video
    if frame_count % 80 == 0:

        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB) # Convert frame to RGB
        pil_image = Image.fromarray(rgb_frame) # Convert frame to PIL
        tensor_frame = transform(pil_image) # Transform frame

        processed_frames.append(tensor_frame)
        grabbed_frames += 1

    frame_count += 1
        
vid.release()
print(f"Grabbed {grabbed_frames} frames from video\n")

video_tensor_batch = torch.stack(processed_frames)

# EXAMPLE CODE for displaying frames
def display_video_frames(tensor_batch, delay=0.5):

    # Define transform
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    for i, frame in enumerate(tensor_batch):

        # Transform
        img = frame * std + mean
        img = torch.clamp(img, 0, 1)
        img = img.permute(1, 2, 0).cpu().numpy()

        # Display image
        plt.imshow(img)
        plt.title(f"Frame {i+1}")
        plt.axis('off')
        plt.pause(delay)
        plt.clf()  # Clear

    plt.close()

#display_video_frames(video_tensor_batch, delay=0.5)
