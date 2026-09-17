# Emotion Detection System

A Streamlit web application that predicts the emotion expressed in a piece of text. The application transforms text with a **TF-IDF vectorizer** and supports two pre-trained classifiers: **Logistic Regression** and **Multinomial Naive Bayes**.

The classifier recognizes six emotions:

- Anger

- Fear

- Joy

- Love

- Sadness

- Surprise

## Demo capabilities

The application provides:

- A text area for entering or pasting text.

- A choice between Logistic Regression and Multinomial Naive Bayes.

- The predicted emotion and confidence score.

- A Plotly probability chart showing the model's class probabilities.

- A Streamlit interface with responsive layout and emotion-specific styling.

## How it works

1. The input text is transformed using the serialized TF-IDF vectorizer in `vectorizer.pkl`.

1. The transformed text is passed to the selected serialized model.

1. Numeric model labels are mapped to emotion names in `app.py`.

1. The predicted emotion and, when available, class probabilities are rendered in the browser.

The label mapping currently used by the application is:

| Numeric label | Emotion |
| --- | --- |
| 0 | Sadness |
| 1 | Joy |
| 2 | Love |
| 3 | Anger |
| 4 | Fear |
| 5 | Surprise |

> The label mapping must remain consistent with the labels used when the model artifacts were trained. If you retrain the models with a different encoding, update `LABEL_MAP` in `app.py`.

## Requirements

- Python 3.9 or newer

- `pip`

- A web browser

The Python packages used by the project are listed in [`requirements.txt`](requirements.txt): Streamlit, pandas, NumPy, scikit-learn, Joblib, NLTK, and Plotly.

## Installation

Clone the repository and move into the project directory:

```bash
git clone https://github.com/Div7anshKushwaha/NLP-Project.git
cd NLP-Project
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell, activate it with:

```
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Run the application

Start Streamlit from the repository root:

```bash
streamlit run app.py
```

Streamlit will print a local URL, normally [`http://localhost:8501`](http://localhost:8501). Open that URL in a browser, enter text, choose a classifier, and select **Predict Emotion**.

The application expects the following files to be in the same directory as `app.py`:

- `vectorizer.pkl`

- `log_model.pkl`

- `nb_model.pkl`

These pre-trained artifacts are included in the repository. If an artifact is missing, the application displays an error identifying the missing file.

## Repository structure

```
.
├── app.py                         # Streamlit application
├── dataset/
│   └── train.txt                 # Semicolon-separated text/emotion dataset
├── notebook/
│   └── project.ipynb             # Exploratory preprocessing and training notebook
├── log_model.pkl                 # Serialized Logistic Regression model
├── nb_model.pkl                  # Serialized Multinomial Naive Bayes model
├── vectorizer.pkl                # Serialized TF-IDF vectorizer
├── requirements.txt              # Python dependencies
├── .devcontainer/
│   └── devcontainer.json         # VS Code/GitHub Codespaces configuration
└── LICENSE
```

## Training workflow

The notebook demonstrates the original modeling workflow:

1. Load `train.txt` as two semicolon-separated columns: `text` and `emotion`.

1. Encode emotion names as integer labels.

1. Normalize text by lowercasing it and removing punctuation, digits, non-ASCII characters, and English stop words.

1. Split the data into training and test sets with an 80/20 split and `random_state=42`.

1. Train Multinomial Naive Bayes and Logistic Regression models on TF-IDF features.

1. Measure accuracy on the held-out test set.

1. Serialize the trained models and vectorizer with Joblib.

To reproduce the notebook workflow, open `notebook/project.ipynb` in Jupyter or VS Code. Because the notebook uses relative file paths, run it with the appropriate working directory or change the data path to `../dataset/train.txt` when executing it from inside the `notebook/` directory.

When exporting new artifacts, use the filenames expected by the application:

```python
import joblib

joblib.dump(logistic_model, "log_model.pkl")
joblib.dump(nb_model, "nb_model.pkl")
joblib.dump(tfidf_vectorizer, "vectorizer.pkl")
```

Place the resulting files in the repository root before starting Streamlit.

## Using the development container

The repository includes a `.devcontainer/devcontainer.json` configuration for Python 3.11. The configuration installs the dependencies, starts Streamlit after attaching to the container, and forwards port `8501`.

If you use GitHub Codespaces or a compatible VS Code Dev Container setup, open the repository in a container and preview the forwarded application port.

## Data format

The training file uses one example per line in the following format:

```
text describing an emotion;emotion_label
```

For example:

```
i feel romantic too;love
```

The dataset contains the six emotion labels listed above. Review the data and obtain any required permissions before redistributing it outside this repository.

## Notes and limitations

- The application loads serialized Joblib files at runtime. Only load model and vectorizer files from a trusted source.

- The prediction quality depends on the training data and preprocessing used to create the checked-in artifacts.

- Confidence values are model probability estimates. They should not be interpreted as guaranteed probabilities of a person's actual emotional state.

- The notebook contains experimentation and evaluation cells rather than a standalone training script. A future improvement would be to move preprocessing and training into a reproducible Python script or scikit-learn pipeline.

## Author

**Divyansh Kushwaha**

- GitHub: [Div7anshKushwaha](https://github.com/Div7anshKushwaha)

- LinkedIn: [Divyansh Kushwaha](https://www.linkedin.com/in/divyansh-kushwaha-603616383)

## License

See [`LICENSE`](LICENSE) for the terms applicable to this project.

## References

[1]: https://streamlit.io/ "Streamlit documentation"

[2]: https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction "scikit-learn text feature extraction documentation"

[3]: https://joblib.readthedocs.io/en/latest/persistence.html "Joblib persistence documentation"

[4]: https://plotly.com/python/ "Plotly Python documentation"

The project uses the technologies described in these references. The repository's own `LICENSE` file governs this project's licensing terms.

[1]
[2]
[3]
[4]

---

Developed by Divyansh Kushwaha.
