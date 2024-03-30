```
# CLIP Model for Text-to-Image Extraction

## Description

This project utilizes the CLIP (Contrastive Language-Image Pre-Training) model architecture to extract text from video and images, converting it into image format using text-to-image techniques.

## Table of Contents

- [CLIP Model](#clip-model)
- [Custom Architecture](#custom-architecture)
- [How to Use](#how-to-use)

## CLIP Model

CLIP (Contrastive Language-Image Pre-Training) is a neural network trained on a variety of (image, text) pairs. It can be instructed in natural language to predict the most relevant text snippet, given an image, without directly optimizing for the task, similarly to the zero-shot capabilities of GPT-2 and 3. We found CLIP matches the performance of the original ResNet50 on ImageNet “zero-shot” without using any of the original 1.28M labeled examples, overcoming several major challenges in computer vision. For more information, visit the [CLIP GitHub Repository](https://github.com/openai/CLIP).

## Custom Architecture

This architecture utilizes the VIT-B 32 model from the CLIP framework to extract text from videos and multiple images. When provided with text input, it retrieves the most similar images from the dataset.

## How to Use

1. Clone the repository:

   ```
   git clone https://github.com/paulsung97/CLIP_model.git
   ```

2. Install the required dependencies:

   ```
   pip install -r requirement.txt
   ```

3. For video input, rename the file as `video.mp4` and place it in the video directory.

4. Run `video_search.py`:

   ```
   python video_search.py
   ```

   This Python script will prompt you to enter text. After entering text, it will extract the most similar image from the video content and save it in the output directory.

5. Run `image_search.py`:

   ```
   python image_search.py
   ```

   This Python script will extract the most similar image from the images directory based on the input text and save it in the output directory.

```
