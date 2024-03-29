import os
import cv2
import clip
import torch
import math
import numpy as np
from PIL import Image
from IPython.core.display import HTML

# Path to your local video file
video_file = r"./videos/video.mp4"

# Number of frames to skip
N = 120

video_frames = []

capture = cv2.VideoCapture(video_file)
fps = capture.get(cv2.CAP_PROP_FPS)

current_frame = 0
while capture.isOpened():
    ret, frame = capture.read()

    if ret == True and frame is not None:
        video_frames.append(Image.fromarray(frame[:, :, ::-1]))
    else:
        break

    current_frame += N
    capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

capture.release()
print(f"Frames extracted: {len(video_frames)}")

# Load the open CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# You can try tuning the batch size for very large videos, but it should usually be OK
batch_size = 256
batches = math.ceil(len(video_frames) / batch_size)

# The encoded features will be stored in video_features
video_features = torch.empty([0, 512], dtype=torch.float16).to(device)

# Process each batch
for i in range(batches):
    print(f"Processing batch {i+1}/{batches}")

    # Get the relevant frames
    batch_frames = video_frames[i*batch_size : (i+1)*batch_size]
    
    # Preprocess the images for the batch
    batch_preprocessed = torch.stack([preprocess(frame) for frame in batch_frames]).to(device)
    
    # Encode with CLIP and normalize
    with torch.no_grad():
        batch_features = model.encode_image(batch_preprocessed)
        batch_features /= batch_features.norm(dim=-1, keepdim=True)

    # Append the batch to the list containing all features
    video_features = torch.cat((video_features, batch_features))

# Create output directory if it doesn't exist
output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)

def search_video(search_query, display_results_count=20):
    # Encode and normalize the search query using CLIP
    with torch.no_grad():
        text_features = model.encode_text(clip.tokenize(search_query).to(device))
        text_features /= text_features.norm(dim=-1, keepdim=True)

    # Compute the similarity between the search query and each frame using the Cosine similarity
    similarities = (100.0 * video_features @ text_features.T)
    values, best_photo_idx = similarities.topk(display_results_count, dim=0)

    for i, frame_id in enumerate(best_photo_idx):
        img_path = os.path.join(output_dir, f"{search_query}_{i+1}.jpg")
        video_frames[frame_id].save(img_path)
        print(f"Saved frame {i+1} to: {img_path}")

# Infinite loop for search
while True:
    search_query = input("Enter your search query (or 'stop' to quit): ")
    if search_query == "stop":
        break
    display_results_count = int(input("Enter the number of photos you want to extract: "))
    search_video(search_query, display_results_count)
