import os
import clip
import torch
from PIL import Image
import hashlib

# Load the open CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Directory containing your images
images_directory = "./images"

# List all image files in the directory
image_files = [f for f in os.listdir(images_directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Encode and normalize all input images using CLIP
image_features = []
for image_file in image_files:
    image_path = os.path.join(images_directory, image_file)
    image = Image.open(image_path)
    preprocessed_image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_feature = model.encode_image(preprocessed_image)
        image_feature /= image_feature.norm(dim=-1, keepdim=True)
        image_features.append((image_path, image_feature))

# Create output directory if it doesn't exist
output_directory = "./output"
os.makedirs(output_directory, exist_ok=True)

# Check if the image directory already contains the image
def image_exists(image_filename):
    existing_names = set()
    for filename in os.listdir(output_directory):
        if filename.endswith('.txt'):
            with open(os.path.join(output_directory, filename), "r") as text_file:
                existing_names.add(text_file.readline().strip().replace("Image: ", ""))
    return image_filename in existing_names

while True:
    # Input text for similarity search
    input_text = input("Enter your search text (or type ':w' to quit): ")
    if input_text == ":w":
        break
    
    with torch.no_grad():
        input_text_feature = model.encode_text(clip.tokenize([input_text]).to(device))
        input_text_feature /= input_text_feature.norm(dim=-1, keepdim=True)

    # Compute similarity scores between the input text and all images
    similarities = [(image_path, (100.0 * image_feature @ input_text_feature.T).item()) for image_path, image_feature in image_features]
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Get the number of existing files in the output directory
    existing_files = [f for f in os.listdir(output_directory) if f.endswith(('.jpg', '.jpeg', '.png'))]
    start_number = len(existing_files) + 1

    # Specify the number of different images to extract
    num_different_images = 1

    # Save the top similar images and their corresponding texts
    print(f"Saving {num_different_images} different images:")
    extracted_images = 0
    for i, (image_path, similarity) in enumerate(similarities):
        image_filename = os.path.basename(image_path)

        if not image_exists(image_filename):
            output_image_path = os.path.join(output_directory, f"{start_number + extracted_images}.jpg")
            output_text_path = os.path.join(output_directory, f"{start_number + extracted_images}.txt")

            # Save the image
            Image.open(image_path).save(output_image_path)
            print(f"Saved {output_image_path} - Similarity: {similarity:.2f} - Image: {image_filename}")

            # Save the text
            with open(output_text_path, "w") as text_file:
                text_file.write(f"Image: {image_filename}\n")
                text_file.write(f"A picture of {input_text}\n")
                print(f"Saved {output_text_path}")

            extracted_images += 1
            if extracted_images >= num_different_images:
                break
        else:
            print(f"Skipping {image_filename} - Already exists in 'output' folder")

    if extracted_images < num_different_images:
        print(f"Not enough different images found. Extracted: {extracted_images}, Desired: {num_different_images}")
