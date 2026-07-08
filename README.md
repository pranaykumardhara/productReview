# productReview

## Overview

`productReview` is a simple sentiment analysis app built with Streamlit. It uses a pre-trained SVM model and TF-IDF vectorizer to classify user feedback as Positive, Neutral, or Negative.

## Repository contents

- `app.py` - Streamlit app for interactive sentiment prediction.
- `model.py` - Training script that builds and serializes the text classifier and vectorizer.
- `demo_data.xlsx` - Example dataset used to train the model.
- `model.pkl` - Serialized trained SVM classifier.
- `tv_transform.pkl` - Serialized TF-IDF vectorizer.

## Requirements

Install Python 3.10+ and the required packages.

```bash
python3 -m pip install streamlit pandas numpy scikit-learn nltk openpyxl
```

## Run the app

Start the Streamlit app from the repository root:

```bash
streamlit run app.py
```

Open the URL shown in the terminal to use the feedback sentiment predictor.

## Retrain the model

If you want to retrain the classifier from `demo_data.xlsx`, run:

```bash
python3 model.py
```

This will recreate `model.pkl` and `tv_transform.pkl` in the repository root.

### NLTK setup for training

If you rerun `model.py` and see missing NLTK resources, install them with:

```bash
python3 -m nltk.downloader stopwords wordnet
```

## Notes

- `app.py` expects `model.pkl` and `tv_transform.pkl` to exist in the same folder.
- The current app performs text preprocessing using the saved TF-IDF transformer and then predicts sentiment from the trained model.
