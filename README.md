# 🖼️ Text-to-Image Generation Model


https://github.com/user-attachments/assets/ed95705f-ccb7-4d43-99be-645fdf83a3c4


📝 **Contents**
- [Introduction](#-introduction)
- [Technology Stack](#-technology-stack)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [How to Run](#-how-to-run)
- [Using the Model](#-using-the-model)
- [Contribution Guidelines](#-contribution-guidelines)
- [Get in Touch](#-get-in-touch)

## 🌟 Introduction
This project implements a **Text-to-Image Generation Model** using TensorFlow and Gradio. The model converts textual descriptions (prompts) into corresponding images by identifying and generating captions for similar images from a dataset.

## 💻 Technology Stack
- **TensorFlow**: For deep learning and model training.
- **Gradio**: For creating the web-based interface.
- **Pandas**: For data manipulation and analysis.
- **Numpy**: For numerical operations.
- **PIL**: For image processing.
- **COCO Dataset**: Used for training and testing the model.

## ✨ Features
- **Image Captioning**: Generate captions for images using a Transformer-based model.
- **Text-to-Image Matching**: Find and display images that match a given textual prompt.
- **Interactive Web Interface**: Use Gradio to interact with the model and visualize results.
- **Model Training and Saving**: Train the model on a subset of the COCO dataset and save the weights.

## 🗂️ Project Structure
- **`coco_dataset/`**: Directory containing the COCO dataset.
  - **`annotations/captions_train2017.json`**: Annotations for training data.
  - **`train2017/`**: Directory containing the training images.
- **`captions_sample.csv`**: Sampled captions and image paths used for training.
- **`vocab_coco.file`**: Serialized vocabulary file created from the tokenizer.
- **`model.weights.h5`**: File containing the saved weights of the trained model.
- **`Text-to-Image-Generation.ipynb`**: Jupyter Notebook containing the code for this project.

## 🚀 How to Run
1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/text-to-image-generator.git
    cd text-to-image-generator
    ```

2. **Install the required dependencies:**
    ```bash
    pip install tensorflow pandas numpy matplotlib pillow tqdm requests scikit-learn gradio
    ```

3. **Prepare the COCO dataset:**
    - Download and place the COCO dataset in the `coco_dataset/` directory.

4. **Run the Jupyter Notebook:**
    - Open `Text-to-Image-Generation.ipynb` and run the cells sequentially.

## 🎨 Using the Model
1. **Generate Images from Text Prompts:**
   - Input a text prompt into the Gradio interface.
   - The model will find the most similar image and display it along with a generated caption.

2. **Save or Load Model Weights:**
   - Train the model and save the weights to `model.weights.h5`.
   - Load saved weights for further training or inference.

## 🤝 Contribution Guidelines
Contributions are welcome! If you'd like to contribute, please follow these steps:

1. **Fork the repository.**
2. **Create a new branch:**
    ```bash
    git checkout -b feature/YourFeatureName
    ```
3. **Make your changes and commit them:**
    ```bash
    git commit -m 'Add YourFeatureName'
    ```
4. **Push to the branch:**
    ```bash
    git push origin feature/YourFeatureName
    ```
5. **Open a pull request** to discuss and merge your changes.

## 📧 Get in Touch
- **Email**: oleg15062005@gmail.com
- **GitHub**: [Oleh Kihichak](https://github.com/OKihichak)
