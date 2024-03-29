import os
import clip
import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# Load the open CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Path to your output directory
output_dir = "./output"

# List to store the cosine similarities, images, and image names for plotting
images = []
image_names = []
cosine_similarities = []

# Preprocess and encode each image in the output directory
for image_name in os.listdir(output_dir):
    if image_name.endswith(".jpg") or image_name.endswith(".png"):
        image_path = os.path.join(output_dir, image_name)
        image = Image.open(image_path)
        processed_image = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(processed_image)
        
        # Normalize the image features
        image_features /= image_features.norm(dim=-1, keepdim=True)

        # Define your text query as the image name without the extension and preprocess and encode it
        text_query = os.path.splitext(image_name)[0]  # Remove the extension from the image name
        text_features = model.encode_text(clip.tokenize([text_query]).to(device))

        # Normalize the text features
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # Compute the similarity between the text features and image features using the Cosine similarity
        cosine_similarity = (100.0 * text_features @ image_features.T).item()
        
        # Append the image, image name, and cosine similarity to the respective lists
        images.append(image)
        image_names.append(text_query)  # Append the image name without the extension
        cosine_similarities.append(cosine_similarity)

# Plotting
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(image_names, cosine_similarities, color='skyblue')
ax.set_xlabel('Cosine Similarity')
ax.set_title('Cosine similarity between text and image features')
ax.set_xlim([0, max(cosine_similarities) + 20])  # Increase the x limit to fit images

# Add cosine similarities inside the bars
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax.text(width - 5, bar.get_y() + bar.get_height()/2.0, '{:.2f}'.format(cosine_similarities[i]), 
            va='center', ha='right', color='black', fontweight='bold')

# Add images beside the bars
for i, (bar, image) in enumerate(zip(bars, images)):
    image = image.resize((50, 50))  # Resize the image
    imagebox = OffsetImage(image, zoom=0.5)
    ab = AnnotationBbox(imagebox, (bar.get_width() + 2, i))  # Set the position of the image
    ax.add_artist(ab)

# Save the plot to a file
plt.savefig(os.path.join(output_dir, 'cosine_similarity_plot.png'), bbox_inches='tight')
plt.show()
