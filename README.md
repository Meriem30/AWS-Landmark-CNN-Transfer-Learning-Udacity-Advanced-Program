# 🏛️ AWS Landmark Classification with CNN & Transfer Learning

This project tackles the challenge of **automatic location detection** through **landmark recognition** in photos. It builds an **end-to-end image classification system** using **Convolutional Neural Networks (CNNs)** and **Transfer Learning**.
This project was completed as part of the **Udacity Advanced AWS Machine Learning Fundamentals Nanodegree Program**.


> #### Reviewer Note for my Project Submission on Udacity Platform
> *"Congratulations on completing the Landmark Classification & Tagging for Social Media 2.0! project
> You have learned how to use deep learning techniques to classify and tag images of landmarks worldwide. You have also built a web app that allows users to upload photos and get predictions from your model. That’s an impressive achievement!
> You should be proud of yourself for completing this challenging and rewarding course. You have demonstrated your skills and knowledge in computer vision, machine learning, and web development. You have also created a portfolio-worthy project that showcases your abilities and creativity.
> I hope you enjoyed this learning journey and found it useful for your personal and professional goals. I encourage you to [keep exploring](https://cs231n.github.io/convolutional-networks/) the fascinating field of artificial intelligence and apply what you have learned to new problems and domains. You have a bright future ahead of you! 😊"*


## 🎯 Overview

This implementation demonstrates:

- Building CNNs from scratch for image classification
- Leveraging transfer learning with pre-trained models (ResNet, VGG, etc.) for improved accuracy
- Data preprocessing and augmentation techniques
- Model training, evaluation, and deployment
- Hyperparameter tuning and model comparison
- Best model exported and ready for production
- Designed for real-world application: simple interface for landmark prediction on new images



## 📁 Project Structure

```
AWS-Landmark-CNN-Transfer-Learning/
│
├── src/                          # Source code
│   ├── model.py                  # Model architectures
│   ├── data.py                   # Data loading utilities 
│   ├── optimizaation.py          # Loss funciton and optimizers
│   ├── train.py                  # Training pipeline
│   ├── predictor.py              # Prediction functions
│   ├── helpers.py                # helper utilities
│   └── transfer.py               # Transfer Learning pipeline
│
├── # Jupyter notebooks
├── cnn_from_scratch.ipynb       # Custom CNN implementation
├── transfer_learning.ipynb      # Transfer learning experiments
├── app.ipynb                    # Application interface for your new landmark images
│
├── # Jupyter notebook HTML pages
├── cnn_from_scratch.html   
├── transfer_learning.html   
├── app.html                
│
├── data/                          # Dataset directory (not included)
│   ├── train/
│   └── test/
│
├── checkpoints/                   # Saved models
│   ├── transfer_exported.pt
│   └── best_model.pth
│
└── README.md                      # Project documentation
```
