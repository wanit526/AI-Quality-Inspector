# 🤖 AI Quality Control Inspector

This project is an end-to-end Deep Learning application that simulates an industrial quality control (QC) inspector.

It uses a Convolutional Neural Network (CNN) trained to classify images, and deploys this model as an interactive web application using Streamlit.

## 🚀 Features

* **Interactive Web App:** Built with Streamlit, allowing users to upload images directly from their browser.
* **Deep Learning Model:** Uses a TensorFlow/Keras CNN model trained on the (TensorFlow Datasets 'beans') to classify images into "Healthy" (ดี) or "Defect" (ชำรุด).
* **Real-time Prediction:** The app processes the uploaded image and provides an instant prediction with a confidence score.
* **End-to-End Workflow:** Demonstrates the full ML lifecycle: data preprocessing, model training (in Google Colab), and deployment (in Streamlit).

## 🔧 Technology Stack

* **AI / ML:** TensorFlow, Keras
* **Web App:** Streamlit
* **Data Processing:** NumPy, Pillow (PIL)
* **Training Environment:** Google Colab (for free GPU access)

## 📸 Demo
![Demo](https://github.com/user-attachments/assets/3e9b95b7-ed17-467f-8000-adaa1ca27967)

## 🎬 How to Run

1.  Clone this repository.
2.  **Download the AI model (`my_ai_model.h5`) from this link:**
    **[➡️ Download Model from Google Drive]([https://link-to-your-google-drive-file](https://drive.google.com/drive/folders/183NGKHnda1mgNDxOZr89m_qGZSvdL7US?usp=sharing))**
3.  Place the downloaded `my_ai_model.h5` file in the same directory.
4.  Install the required libraries:
    ```bash
    python -m pip install streamlit tensorflow pillow
    ```
5.  Run the Streamlit app:
    ```bash
    streamlit run ai_inspector_app.py
    ```
