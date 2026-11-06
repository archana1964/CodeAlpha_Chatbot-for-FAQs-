# FAQ Chatbot

A simple FAQ chatbot built with Flask, NLTK, and scikit-learn. It preprocesses questions and uses TF-IDF with cosine similarity to select the closest stored FAQ answer.

## 1. Requirements
- Python 3.9 or later
- Internet connection for the first-time package installation

## 2. Create and activate a virtual environment (Windows PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

## 3. Install dependencies

Run this command from the project folder (the folder containing `requirements.txt`):

```powershell
python -m pip install -r requirements.txt
```

## 4. Run the application

```powershell
python app.py
```

Open this address in your browser:

`http://127.0.0.1:5000`

## 5. Add or edit FAQs

Open `faqs.json`. Each FAQ object must contain:
- `question`: the main question
- `answer`: the chatbot response
- `alternatives`: optional list of other ways users may ask the same question

Example:

```json
{
  "question": "How can I contact support?",
  "alternatives": ["What is the support contact?", "How do I reach the team?"],
  "answer": "You can contact the support team by email."
}
```

After editing the JSON file, restart the Flask application so the FAQ vectors are rebuilt.

## How it works

1. The user enters a question in the web chat interface.
2. Flask receives the question through `/ask`.
3. NLTK preprocessing lowercases the text, tokenizes it, removes common stopwords, and applies Porter stemming.
4. scikit-learn converts the stored FAQ questions and user question into TF-IDF vectors.
5. Cosine similarity finds the closest FAQ question.
6. If the best score is at least 0.18, the corresponding answer is returned; otherwise, a fallback response is displayed.

The 0.18 threshold is a starting value, not a measured accuracy guarantee. Adjust it after testing with your own questions.
