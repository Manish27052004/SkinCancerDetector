# 🧬 Melanoma Skin Cancer Detection AI

This repository contains the neural network architecture, training scripts, and web UI for an advanced skin cancer classification system.

## 👥 Collaboration Structure
*   **Person 1 (Developer/UI):** Handles the neural network architecture (`models/model_builder.py`) and Streamlit web frontend (`app.py`).
*   **Person 2 (Trainer/Evaluator):** Handles downloading the image dataset, running the training sequence locally, evaluating the metrics, and sending the final trained `.h5` model to Person 1.

---

## 🚀 Setup Instructions for Person 2 (The Trainer)

Hello! If you have just cloned this repository, your job is to set up your local environment, acquire the dataset, and train the model. 

### 🤖 AI Setup Prompt (Copy and Paste to your AI Assistant)
*If you are using Antigravity, ChatGPT, or another AI assistant to help you set this up, copy the entire prompt below and paste it into your chat:*

> **Prompt for AI Assistant:**
> "I am Person 2 (the Model Trainer) working on a collaborative skin cancer detection project. I have just cloned the GitHub repository to my local Windows machine. I need your help to instantly set up my environment and train the model.
> 
> **Here are the strict technical constraints you must follow:**
> 1. The project uses TensorFlow natively on Windows, which means you MUST help me install exactly **Python 3.10**. Do not use Python 3.11+.
> 2. Create a virtual environment named `env` in the root of the project.
> 3. Activate the environment and run `pip install -r requirements.txt`.
> 4. The `dataset/` folder is currently empty. Please guide me exactly how/where to download a Melanoma image dataset (like ISIC from Kaggle).
> 5. You must write a Python script (or use bash) to unzip my downloaded dataset and automatically arrange the `.jpg` pictures into the exact `dataset/train/Benign`, `dataset/val/Melanoma`, etc. structure required by `image_dataset_from_directory()`.
> 6. Once the images are in place, run `python training/train_model.py`.
> 7. When training completes, help me locate the `melanoma_detector_model.h5` file inside `outputs/models/` so I can send it back to my partner."

---

## 📁 Repository Structure
*   `app.py`: The Streamlit web application frontend.
*   `models/model_builder.py`: Defines the EfficientNetB0 neural network.
*   `training/train_model.py`: The master script loop that loads data, trains, and saves the `.h5` model.
*   `prediction/predict_image.py`: Command-line inference script.
*   `utils/`: Contains preprocessing functions and graphing metrics.
*   `dataset/`: **[GIT IGNORED]** You must place your raw images in folders here before training.
*   `outputs/`: **[GIT IGNORED]** Contains the saved model weights (`.h5`) and generated graph images after training completes.
