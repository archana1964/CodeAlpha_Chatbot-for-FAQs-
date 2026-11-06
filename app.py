from flask import Flask, render_template, request, jsonify
import json
import os
import re
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAQ_FILE = os.path.join(BASE_DIR, "faqs.json")
stemmer = PorterStemmer()

# A small built-in stopword list keeps the project easy to run:
# no NLTK corpus download is required.
STOP_WORDS = {
    "a", "an", "the", "is", "are", "am", "was", "were", "be", "been",
    "being", "to", "of", "in", "on", "at", "for", "from", "with", "by",
    "and", "or", "but", "if", "then", "this", "that", "these", "those",
    "it", "its", "i", "you", "we", "they", "he", "she", "my", "your",
    "our", "their", "me", "do", "does", "did", "can", "could", "would",
    "should", "will", "what", "when", "where", "why", "how", "which"
}

def preprocess(text):
    """Lowercase, remove punctuation, tokenize, remove stopwords and stem."""
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    cleaned = [
        stemmer.stem(token)
        for token in tokens
        if token not in STOP_WORDS and len(token) > 1
    ]
    return " ".join(cleaned)

def load_faqs():
    with open(FAQ_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)
    # Each FAQ has one main question, optional alternative questions, and an answer.
    questions, answers = [], []
    for item in data:
        all_questions = [item["question"]] + item.get("alternatives", [])
        for question in all_questions:
            questions.append(preprocess(question))
            answers.append(item["answer"])
    return questions, answers

FAQ_QUESTIONS, FAQ_ANSWERS = load_faqs()
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
FAQ_MATRIX = vectorizer.fit_transform(FAQ_QUESTIONS)

def find_answer(user_question):
    cleaned_question = preprocess(user_question)
    if not cleaned_question.strip():
        return {
            "answer": "Please enter a question so I can help you.",
            "confidence": 0.0,
            "matched": False
        }

    user_vector = vectorizer.transform([cleaned_question])
    scores = cosine_similarity(user_vector, FAQ_MATRIX).flatten()
    best_index = int(scores.argmax())
    best_score = float(scores[best_index])

    # Adjust this threshold if you want the chatbot to be stricter or more flexible.
    threshold = 0.18
    if best_score < threshold:
        return {
            "answer": "Sorry, I couldn't find a matching FAQ. Please try rephrasing your question or contact the support team.",
            "confidence": round(best_score * 100, 1),
            "matched": False
        }

    return {
        "answer": FAQ_ANSWERS[best_index],
        "confidence": round(best_score * 100, 1),
        "matched": True
    }

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Please type a question."}), 400
    return jsonify(find_answer(question))

if __name__ == "__main__":
    app.run(debug=True)
