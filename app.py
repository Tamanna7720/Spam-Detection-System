from flask import Flask, render_template, request, jsonify
import pandas as pd
import re
import os
import joblib
from datetime import datetime, timezone

try:
    from pymongo import MongoClient
except ImportError:
    MongoClient = None

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

app = Flask(__name__)

DATA_URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
MODEL_FILE = "spam_model.pkl"
VECTORIZER_FILE = "tfidf_vectorizer.pkl"
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "spamguard")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "messages")

mongo_client = None
messages_collection = None
in_memory_history = []


def format_timestamp(value):
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    return str(value) if value else "N/A"


def connect_to_mongodb():
    global mongo_client, messages_collection

    if MongoClient is None:
        print("MongoDB driver not installed. Install pymongo to enable database storage.")
        mongo_client = None
        messages_collection = None
        return False

    try:
        mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
        mongo_client.admin.command("ping")
        db = mongo_client[MONGO_DB_NAME]
        messages_collection = db[MONGO_COLLECTION]
        print(f"MongoDB connected: {MONGO_URI}")
        return True
    except Exception as exc:
        mongo_client = None
        messages_collection = None
        print(f"MongoDB connection failed: {exc}")
        return False


def save_prediction(message, result, spam_probability, ham_probability):
    entry = {
        "message": message,
        "result": result,
        "spam_probability": round(float(spam_probability), 2),
        "ham_probability": round(float(ham_probability), 2),
        "created_at": datetime.now(timezone.utc)
    }

    if messages_collection is not None:
        try:
            messages_collection.insert_one(entry)
            return True
        except Exception as exc:
            print(f"MongoDB save failed: {exc}")

    in_memory_history.append(entry)
    if len(in_memory_history) > 20:
        in_memory_history.pop(0)
    return True


def get_recent_history(limit=5):
    if messages_collection is not None:
        try:
            docs = list(messages_collection.find({}).sort("created_at", -1).limit(limit))
            cleaned = []
            for doc in docs:
                cleaned.append({
                    "message": doc.get("message", ""),
                    "result": doc.get("result", "SAFE"),
                    "spam_probability": doc.get("spam_probability", 0),
                    "ham_probability": doc.get("ham_probability", 0),
                    "created_at": format_timestamp(doc.get("created_at"))
                })
            return cleaned
        except Exception as exc:
            print(f"MongoDB history fetch failed: {exc}")

    history = sorted(in_memory_history, key=lambda item: item.get("created_at", datetime.now(timezone.utc)), reverse=True)[:limit]
    return [{
        "message": item.get("message", ""),
        "result": item.get("result", "SAFE"),
        "spam_probability": item.get("spam_probability", 0),
        "ham_probability": item.get("ham_probability", 0),
        "created_at": format_timestamp(item.get("created_at"))
    } for item in history]


def get_summary_stats():
    if messages_collection is not None:
        try:
            total = messages_collection.count_documents({})
            spam = messages_collection.count_documents({"result": "SPAM"})
            safe = messages_collection.count_documents({"result": "SAFE"})
            return {
                "total": total,
                "spam": spam,
                "safe": safe,
                "recent": total,
                "database_connected": True
            }
        except Exception as exc:
            print(f"MongoDB stats fetch failed: {exc}")

    total = len(in_memory_history)
    spam = sum(1 for item in in_memory_history if item.get("result") == "SPAM")
    safe = sum(1 for item in in_memory_history if item.get("result") == "SAFE")
    return {
        "total": total,
        "spam": spam,
        "safe": safe,
        "recent": total,
        "database_connected": False
    }


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " URL ", text)
    text = re.sub(r"\S+@\S+", " EMAIL ", text)
    text = re.sub(r"\d+", " NUMBER ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def train_model():
    print("Downloading SMS Spam dataset...")
    df = pd.read_csv(
        DATA_URL,
        sep="\t",
        header=None,
        names=["label", "message"]
    )

    df = df.dropna().drop_duplicates()
    df["message"] = df["message"].apply(clean_text)
    df["label"] = df["label"].map({"ham": 0, "spam": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        df["message"],
        df["label"],
        test_size=0.20,
        random_state=42,
        stratify=df["label"]
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=7000,
        sublinear_tf=True
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = MultinomialNB(alpha=0.5)
    model.fit(X_train_tfidf, y_train)

    accuracy = accuracy_score(
        y_test,
        model.predict(X_test_tfidf)
    )

    joblib.dump(model, MODEL_FILE)
    joblib.dump(vectorizer, VECTORIZER_FILE)

    print(f"Model trained successfully. Accuracy: {accuracy * 100:.2f}%")
    return accuracy


def load_or_train():
    if os.path.exists(MODEL_FILE) and os.path.exists(VECTORIZER_FILE):
        return (
            joblib.load(MODEL_FILE),
            joblib.load(VECTORIZER_FILE)
        )

    accuracy = train_model()
    return (
        joblib.load(MODEL_FILE),
        joblib.load(VECTORIZER_FILE)
    )


model, vectorizer = load_or_train()
connect_to_mongodb()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()

    if not message:
        return jsonify({
            "success": False,
            "error": "Please enter an email or SMS message."
        }), 400

    cleaned = clean_text(message)
    features = vectorizer.transform([cleaned])

    prediction = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]

    spam_probability = float(probabilities[1] * 100)
    ham_probability = float(probabilities[0] * 100)

    if prediction == 1:
        result = "SPAM"
        icon = "🚨"
        description = "This message looks suspicious and may be unwanted."
    else:
        result = "SAFE"
        icon = "✓"
        description = "This message looks like a normal message."

    database_saved = save_prediction(
        message,
        result,
        spam_probability,
        ham_probability
    )

    return jsonify({
        "success": True,
        "result": result,
        "icon": icon,
        "description": description,
        "spam_probability": round(spam_probability, 2),
        "ham_probability": round(ham_probability, 2),
        "database_connected": database_saved
    })


@app.route("/model-info")
def model_info():
    return jsonify({
        "algorithm": "Multinomial Naive Bayes",
        "features": "TF-IDF",
        "technique": "Natural Language Processing",
        "dataset": "SMS Spam Collection",
        "classes": ["Spam", "Ham"]
    })


@app.route("/history")
def history():
    entries = get_recent_history(limit=5)
    return jsonify({
        "success": True,
        "history": entries,
        "database_connected": messages_collection is not None
    })


@app.route("/stats")
def stats():
    return jsonify({
        "success": True,
        "stats": get_summary_stats()
    })


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("        SPAMGUARD AI - SPAM DETECTION SYSTEM")
    print("=" * 60)
    print("Open: http://127.0.0.1:5000")
    print("=" * 60 + "\n")
    app.run(debug=True)
