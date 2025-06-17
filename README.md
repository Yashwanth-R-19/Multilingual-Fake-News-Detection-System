# Multilingual Fake News Detection System

This is a Flask-based Fake News Detection web application that uses a RoBERTa-based sequence classification model to predict whether a given news statement is likely to be True or False.
The application is designed with bilingual support (English and Tamil) and a clean, responsive interface for an intuitive user experience.

## Key Features

* Real-time fake news detection using a fine-tuned RoBERTa model.
* Bilingual display: English and Tamil.
* Fully responsive web interface with modern styling.
* Instant prediction without page reloads.

## Note:

* **Model Folder Not Included**:  The saved_model folder is NOT included in this repository because of its large size.
This folder contains the fine-tuned RoBERTa model files required to run the application.

To use this application:

* You must train and save the model to the ./saved_model path.
  
## Technologies Used

* Python
* Flask
* PyTorch
* Hugging Face Transformers
* HTML, CSS, JavaScript
