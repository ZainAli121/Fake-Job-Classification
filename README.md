# Fake Job Classification Using NLP and Machine Learning
This project is about classifying real vs. fake job postings using Natural Language Processing (NLP) and machine learning. The system takes the text from a job advertisement and predicts whether it is legitimate or fraudulent based on its content.

## Dataset Used

I used the Recruitment Scam Dataset from Kaggle.

Dataset link:  
https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction

This dataset contains **17,880 real job postings** collected from online platforms.

### Class Distribution
- Real job postings: 17,014
- Fake job postings: 866

## Model Used

I used a Linear Support Vector Machine (SVM) for this project.

- Text features were extracted using TF-IDF vectorization
- The model was trained to distinguish between real and fake job postings
- Training was done on an 80/20 train-test split

## Tools and Libraries Used

- Python
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- NLTK

## How the Project Works

1. Load and preprocess the job posting text data
2. Combine relevant text fields (title, description, requirements)
3. Apply TF-IDF to convert text into numerical features
4. Train a Linear SVM classifier
5. Evaluate the model on test data
6. Analyze results using accuracy, confusion matrix, and classification report

## Results
- **Model Accuracy**: 99.21%

The results show that the model performs very well in identifying fake job postings.

## Purpose of This Project

This project was created for learning and practice.

- It helps in understanding:
- Text classification with NLP
- Feature extraction using TF-IDF
- Machine learning with SVM
- Handling imbalanced datasets
- Real-world fraud detection applications


## Conclusion
This project demonstrates how traditional machine learning models combined with NLP techniques can effectively detect fake job postings.
With proper text preprocessing and feature engineering, the model achieves high accuracy and can help protect job seekers from scams.

Thank you for reading.

