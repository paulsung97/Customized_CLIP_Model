Sure, here is the content formatted for GitHub:

```markdown
## Architecture for Extracting Text-to-Image using CLIP Model

### 1. What is CLIP?
[CLIP (Contrastive Language-Image Pre-Training)](https://github.com/openai/CLIP) is a neural network trained on a variety of (image, text) pairs. It can be instructed in natural language to predict the most relevant text snippet, given an image, without directly optimizing for the task, similarly to the zero-shot capabilities of GPT-2 and 3. We found CLIP matches the performance of the original ResNet50 on ImageNet "zero-shot" without using any of the original 1.28M labeled examples, overcoming several major challenges in computer vision.

### 2. Custom Architecture
This architecture utilizes the VIT-B 32 model from the CLIP model to extract text from videos and multiple images. When provided with input text, it retrieves the most similar image corresponding to that text.

### 3. How to Use?

1. Clone the repository:
    ```
    $ git clone https://github.com/paulsung97/CLIP_model.git
    ```

2. Install the required dependencies:
    ```
    $ pip install -r requirement.txt
    ```

3. For videos, rename the file to `video.mp4` and place it in the video directory.

4. Run the script for video search:
    ```
    $ python video_search.py
    ```
   Enter the text, and the script will extract the most similar image from the video content and save it to the output file.

5. Run the script for image search:
    ```
    $ python image_search.py
    ```
   The script will extract the most similar image from the images directory based on the input text and save it to the output file.
```
```
