# Indian Currency Notes Classifier

A deep learning-based image classifier to identify Indian currency notes using transfer learning with ResNet-18. Trained on a publicly available dataset and deployed via a user-friendly Streamlit web app.

---
## Demo Video 

https://github.com/user-attachments/assets/90639aee-b63e-4051-8244-b8508c3ffe2d

---

##  Deployed Link

https://currency-note-classifier-bysaumya.streamlit.app/

---

##  Project Overview

This project demonstrates the use of **transfer learning** with a pretrained ResNet-18 model to classify Indian currency notes into 7 denominations:

> ₹10, ₹20, ₹50, ₹100, ₹200, ₹500, ₹2000

The trained model is integrated into an interactive web app using **Streamlit**.

---

##  Tech Stack

- Python
- PyTorch
- Torchvision
- ResNet-18 (pretrained)
- Streamlit
- gdown (for model download)
- Kaggle Datasets

---

##  Dataset

- Source: [Indian Currency Classification Dataset – Kaggle](https://www.kaggle.com/datasets/najiaboo/indiancurrency-for-classification)
- Contains images of Indian currency notes organized into train and validation folders.
- Images were resized and normalized; data augmentation was applied during training.

---

##  Model Details

- **Base Model**: ResNet-18 pretrained on ImageNet
- **Modified Head**: Custom fully connected layers (Linear → ReLU → Dropout → Linear)
- **Loss**: CrossEntropyLoss
- **Optimizer**: Adam
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score

---

##  Web App Features

- Upload an image of a currency note
- Get the predicted denomination
- Download model dynamically from Google Drive to reduce repo size


---

##  Run Locally

```bash
git clone https://github.com/your-username/currency-classifier
cd currency-classifier
pip install -r requirements.txt
streamlit run app.py
