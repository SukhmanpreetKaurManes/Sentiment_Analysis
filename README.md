# Amazon Alexa Review Sentiment Analysis

## Project Overview

This project uses Natural Language Processing (NLP) and deep learning to classify Amazon Alexa product reviews as **Positive** or **Negative**. It explores customer feedback and builds a sentiment classification model using a hybrid CNN + Bidirectional LSTM (BiLSTM) architecture.

A Streamlit web application allows users to enter individual reviews or upload a CSV file to predict sentiment.

## Objectives

- Analyze customer reviews to understand sentiment patterns.
- Perform exploratory data analysis (EDA) on review ratings and feedback.
- Preprocess text data and prepare it for machine learning and deep learning models.
- Handle class imbalance to improve the detection of negative reviews.
- Compare traditional machine learning models with a deep learning approach.
- Deploy the trained model through an interactive Streamlit application.

## Dataset

The project uses an Amazon Alexa reviews dataset containing the following columns:

- `rating`: Customer rating.
- `date`: Date of the review.
- `variation`: Alexa product variation.
- `verified_reviews`: Text of the customer review.
- `feedback`: Sentiment label.

The original dataset contains 3,150 records.

The `feedback` column is the target variable:
- `0` — Negative
- `1` — Positive

The dataset is imbalanced, with significantly more positive reviews than negative reviews. Duplicate reviews were removed during data preparation.

**Note:** The model predicts sentiment from review text, not from the rating, to avoid target leakage.

## Technologies Used

- **Python** — Programming language
- **Pandas and NumPy** — Data manipulation
- **Matplotlib** — Data visualization
- **Scikit-learn** — Data splitting, evaluation, and traditional machine learning models
- **TensorFlow / Keras** — Deep learning model development
- **NLTK / Regular Expressions** — Text preprocessing, where applicable
- **Streamlit** — Interactive web application

## Project Workflow

1. **Data Collection and Understanding** — Load the dataset and inspect its structure.
2. **Exploratory Data Analysis** — Examine ratings, sentiment distribution, product variations, and review lengths.
3. **Data Cleaning** — Check missing values and remove duplicate reviews.
4. **Text Preparation** — Tokenize review text and convert it into padded sequences.
5. **Class Imbalance Handling** — Oversample the minority class within the training data.
6. **Model Development** — Train and compare traditional machine learning models and deep learning models.
7. **Model Evaluation** — Evaluate performance using accuracy, precision, recall, F1-score, and confusion matrices.
8. **Model Saving** — Save the trained model and tokenizer for inference.
9. **Deployment** — Build an interactive Streamlit app for single-review and batch predictions.

## Models Evaluated

Traditional machine learning models:
- Logistic Regression
- Balanced Logistic Regression
- Balanced Linear Support Vector Machine (Linear SVM)
- Multinomial Naive Bayes
- Random Forest
- XGBoost

Deep learning models:
- Bidirectional LSTM (BiLSTM)
- CNN + BiLSTM hybrid model

The balanced Linear SVM performed strongly among the traditional models. The CNN + BiLSTM model was the best-performing candidate among the deep learning experiments based on the held-out test results.

## CNN + BiLSTM Model

The hybrid architecture combines:

- **Embedding Layer:** Converts tokens into dense vector representations.
- **CNN Layers:** Capture local patterns and important phrases in review text.
- **Global Max Pooling:** Extracts prominent features from convolutional outputs.
- **Bidirectional LSTM:** Learns sequential context from both directions.
- **Dense Layers and Dropout:** Combine learned features and help reduce overfitting.
- **Sigmoid Output Layer:** Produces a probability for positive sentiment.

Class imbalance was addressed through oversampling of the minority class in the training data.

## Results

The final CNN + BiLSTM model was evaluated on a held-out test set using a decision threshold of 0.40, selected through validation threshold tuning.

| Metric | Score |
|---|---:|
| Accuracy | 93.35% |
| Positive-class Precision | 95.63% |
| Positive-class Recall | 97.14% |
| Positive-class F1-score | 96.38% |
| Macro F1-score | 78.01% |
| Negative-class F1-score | 59.65% |
| Negative-class Recall | 54.84% |

### Classification Report

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Negative | 0.65 | 0.55 | 0.60 | 31 |
| Positive | 0.96 | 0.97 | 0.96 | 315 |
| Macro Average | 0.81 | 0.76 | 0.78 | 346 |
| Weighted Average | 0.93 | 0.93 | 0.93 | 346 |

The model achieved 93.35% accuracy and a Macro F1-score of 78.01%. It identified approximately 55% of negative reviews, highlighting the challenge of detecting the minority class in an imbalanced dataset.

The test set contained 346 reviews, including 31 negative and 315 positive reviews. Because the number of negative test examples is small, minority-class performance estimates may vary considerably.

## Streamlit Web Application

The application supports:

- Entering a review and predicting its sentiment.
- Uploading a CSV file for batch predictions.
- Viewing sentiment probabilities and confidence estimates.
- Visualizing prediction results.
- Downloading predictions as a CSV file.

### Run the Application Locally

1. Clone or download this repository.
2. Open the project folder in VS Code.
3. Create and activate a virtual environment.
4. Install the required dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

5. Ensure the trained model and tokenizer are saved at the paths expected by `app.py`.
6. Start the Streamlit application:

   ```bash
   streamlit run app.py
   ```

7. Open the local URL displayed in the terminal.


## Key Learnings

- Performing exploratory data analysis on textual data.
- Preparing text for machine learning and deep learning models.
- Understanding and addressing imbalanced classification.
- Comparing traditional machine learning algorithms with deep learning architectures.
- Evaluating classification models beyond accuracy.
- Saving and reusing trained models and tokenizers.
- Building an interactive NLP application with Streamlit.

## Future Improvements

- Expand the dataset to include more negative reviews.
- Improve the detection of short and ambiguous negative reviews.
- Evaluate performance on additional unseen reviews.
- Explore pretrained language models such as BERT.
- Deploy the application online for public access.

## Conclusion

This project demonstrates an end-to-end sentiment analysis workflow, from exploratory data analysis and model development to deployment through Streamlit. It also highlights the importance of handling class imbalance and evaluating minority-class performance instead of relying on accuracy alone.
