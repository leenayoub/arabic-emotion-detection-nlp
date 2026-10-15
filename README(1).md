# Emotion Detection from Text Using NLP and Machine Learning Approaches - Phase 2

## Project Overview

This project performs multiclass emotion classification on Arabic social-media text from the ArPanEmo dataset. It includes text preprocessing, TF-IDF feature extraction, a majority baseline, Logistic Regression, fine-tuned AraBERT, model evaluation, confusion matrices, error analysis, and a Streamlit graphical user interface (GUI) for testing the trained models on new Arabic text.

## Team Members

- Leen Mohammed - 202310323
- Selena Kalbouneh - 202310672
- Doha Alnaeimat - 202310679

Instructor: Dr. Mohammed Atir

## Project Files

```text
NLP_Phase2/
|
|-- phase2_complete_pipeline.py
|-- app.py
|-- requirements.txt
|-- README.md
|
|-- ArPanEmo_train_clean.csv
|-- ArPanEmo_test_clean.csv
|-- ArPanEmo_train_split.csv
|-- ArPanEmo_validation_split.csv
|
|-- phase2_complete_summary.json
|
|-- logistic_regression_validation_report.csv
|-- logistic_regression_test_report.csv
|-- logistic_regression_validation_errors.csv
|-- logistic_regression_test_errors.csv
|-- logistic_regression_validation_confusion.png
|-- logistic_regression_test_confusion.png
|
|-- arabert_validation_report.csv
|-- arabert_test_report.csv
|-- arabert_validation_errors.csv
|-- arabert_test_errors.csv
|-- arabert_validation_confusion.png
|-- arabert_test_confusion.png
|
|-- phase2_complete_outputs/
    |-- tfidf_logistic_regression.joblib
    |-- arabert_best_model/
```

Remove copy numbers automatically added to filenames. For example, rename `phase2_complete_pipeline(4).py` to `phase2_complete_pipeline.py`, `ArPanEmo_train_clean(6).csv` to `ArPanEmo_train_clean.csv`, and `arabert_test_report(1).csv` to `arabert_test_report.csv`.

## Source Code

`phase2_complete_pipeline.py` is the main executable file. It includes dataset inspection, completed preprocessing, TF-IDF extraction, the baseline, Logistic Regression, AraBERT, validation/test evaluation, reports, confusion matrices, error files, and summary generation.

## Cleaned Data Files

| File | Records | Purpose |
|---|---:|---|
| `ArPanEmo_train_clean.csv` | 1,419 | Complete cleaned training data before splitting |
| `ArPanEmo_train_split.csv` | 1,135 | Training both NLP models |
| `ArPanEmo_validation_split.csv` | 284 | Preliminary evaluation and model comparison |
| `ArPanEmo_test_clean.csv` | 250 | Final held-out evaluation |

The filenames must be written exactly as shown because the pipeline loads them automatically when the original raw CSV files are unavailable. Each file contains `number`, `post`, `label`, and `clean_text`.

## Completed Preprocessing

The preprocessing stage:

- Removes unavailable, empty, dots-only, numeric-only, and invalid texts.
- Removes HTML tags.
- Replaces URLs with `URL` and mentions with `USER`.
- Removes `#` while preserving the hashtag word.
- Removes Arabic diacritics and tatweel.
- Normalizes Arabic letter variations.
- Reduces excessive character repetition.
- Preserves negation, emojis, expressive punctuation, and Arabic/English words.
- Removes conflicting labels and duplicate cleaned texts.
- Removes train-test overlap to prevent data leakage.

The final cleaned data contain 1,419 training records and 250 test records. The cleaned training data are divided into 1,135 training and 284 validation records.

## Feature Extraction

The Logistic Regression approach combines:

- Word TF-IDF with unigrams and bigrams: 3,037 features.
- Character TF-IDF with 3-5 character n-grams: 17,504 features.
- Total combined features: 20,541.

TF-IDF is fitted only on `ArPanEmo_train_split.csv`. Validation and test texts are transformed with the same fitted extractor. AraBERT does not use TF-IDF and receives tokenized cleaned text directly.

## Implemented Approaches

1. Majority-class baseline: reference model that predicts the most frequent training class.
2. TF-IDF + Logistic Regression: multiclass classifier trained on combined word and character features.
3. Fine-tuned AraBERT: `aubmindlab/bert-base-arabertv02` fine-tuned for three epochs.

The baseline is only a reference and is not counted as one of the two main NLP approaches.

## Graphical User Interface (GUI)

A graphical user interface was implemented in `app.py` using Streamlit. It allows the user to enter new Arabic text and select one of the two trained approaches:

- TF-IDF + Logistic Regression.
- Fine-tuned AraBERT.

The interface applies the same text-cleaning rules used during training, loads the previously saved model without retraining it, and displays the predicted emotion in English and Arabic. It also shows the model confidence, cleaned input text, and the five highest-probability predictions.

The GUI expects the saved models at the following locations:

```text
phase2_complete_outputs/tfidf_logistic_regression.joblib
phase2_complete_outputs/arabert_best_model/
```

The displayed confidence represents the model's estimated probability and is not a guarantee that the prediction is correct.

## Result Files

### Final Summary

`phase2_complete_summary.json` contains dataset counts, feature counts, preprocessing information, and metrics for the baseline, Logistic Regression, and AraBERT.

### Classification Reports

```text
logistic_regression_validation_report.csv
logistic_regression_test_report.csv
arabert_validation_report.csv
arabert_test_report.csv
```

Each report contains precision, recall, F1-score, and support for every emotion, together with accuracy, macro average, and weighted average.

### Error-Analysis Files

```text
logistic_regression_validation_errors.csv
logistic_regression_test_errors.csv
arabert_validation_errors.csv
arabert_test_errors.csv
```

These contain incorrectly classified records with the original post, cleaned text, correct label, and predicted label.

### Confusion Matrices

```text
logistic_regression_validation_confusion.png
logistic_regression_test_confusion.png
arabert_validation_confusion.png
arabert_test_confusion.png
```

Rows represent true labels, columns represent predicted labels, diagonal values are correct predictions, and off-diagonal values are errors.

## Installation and Execution

Install the required packages:

```powershell
pip install -r requirements.txt
```

Place the Python script and four cleaned CSV files in the same folder, then run:

```powershell
python phase2_complete_pipeline.py
```

The program creates `phase2_complete_outputs`, containing generated datasets, features, models, reports, errors, confusion matrices, and `phase2_complete_summary.json`. Internet access is required when AraBERT is downloaded for the first time.

After the trained models have been generated, start the GUI from the project folder using:

```powershell
python -m streamlit run app.py
```

The browser will open the application automatically. If it does not open, use the local URL displayed in the terminal, usually `http://localhost:8501`.

## Final Results

### Validation

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Majority baseline | 13.03% | 2.31% | 3.00% |
| TF-IDF + Logistic Regression | **50.00%** | **47.42%** | **48.69%** |
| AraBERT | 46.13% | 42.16% | 44.12% |

### Held-out Test

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| TF-IDF + Logistic Regression | 46.40% | **35.81%** | **45.64%** |
| AraBERT | **49.20%** | 35.80% | 44.63% |

Logistic Regression achieved the strongest validation performance. AraBERT achieved higher test accuracy, while Logistic Regression achieved slightly higher test Macro F1 and Weighted F1.

## Main Limitation

Most redistributed Twitter records contain only an ellipsis instead of their original text. After cleaning, only 1,669 usable records remained, including only one usable Neutral training example. Results for Neutral and other classes with very small support must therefore be interpreted cautiously.
