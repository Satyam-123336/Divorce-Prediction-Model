from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

app = Flask(__name__)

# Vercel has a read-only filesystem everywhere except /tmp.
# We read the bundled model at startup; retrain saves to /tmp.
_BUNDLE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(_BUNDLE_DIR, "divorce_model.pkl")   # read (bundled)
MODEL_SAVE   = "/tmp/divorce_model.pkl"                          # write (Vercel /tmp)
CSV_PATH     = os.path.join(_BUNDLE_DIR, "divorce.csv")

QUESTIONS = [
    "When one of us apologizes when our discussions go in a bad direction, the issue does not extend.",
    "I know we can ignore our differences, even if things get hard sometimes.",
    "When we need it, we can take our discussions from the beginning and correct it.",
    "When I argue with my partner, it will eventually work out.",
    "The time I spend with my partner is special for us.",
    "We don't have time at home as partners.",
    "We are like two strangers who share the same environment at home rather than family.",
    "I enjoy our holidays with my partner.",
    "I enjoy traveling with my partner.",
    "My partner and most of our goals are common.",
    "I think that one day in the future, when I look back, I see that my partner and I are in harmony.",
    "My partner and I have similar values in terms of personal freedom.",
    "My partner and I have similar entertainment.",
    "Most of our goals for people (children, friends, etc.) are the same.",
    "Our dreams of living with my partner are similar and harmonious.",
    "We're compatible about what love should be.",
    "We share the same views about being happy in life.",
    "My partner and I have similar ideas about how marriage should be.",
    "My partner and I have similar ideas about how roles should be in marriage.",
    "My partner and I have similar values in trust.",
    "I know exactly what my partner likes.",
    "I know how my partner wants to be taken care of when they're sick.",
    "I know my partner's favorite food.",
    "I can tell you what kind of stress my partner is facing in life.",
    "I have knowledge of my partner's inner world.",
    "I know my partner's basic concerns.",
    "I know what my partner's current sources of stress are.",
    "I know my partner's hopes and wishes.",
    "I know my partner very well.",
    "I know my partner's friends and their social relationships.",
    "I feel aggressive when I argue with my partner.",
    "When discussing with my partner, I usually use expressions such as 'you always' or 'you never'.",
    "I can use negative statements about my partner's personality during discussions.",
    "I can use offensive expressions during discussions.",
    "I can insult during discussions.",
    "I can be humiliating when we argue.",
    "My argument with my partner is not calm.",
    "I hate my partner's way of bringing things up.",
    "Fights often occur suddenly.",
    "We're just starting a fight before I know what's going on.",
    "When I talk to my partner about something, my calm suddenly breaks.",
    "When I argue with my partner, I suddenly snap and stop saying a word.",
    "I'm mostly trying to calm the environment a little bit.",
    "Sometimes I think it's good for me to leave home for a while.",
    "I'd rather stay silent than argue with my partner.",
    "Even if I'm right in the argument, I avoid upsetting the other side.",
    "When I argue with my partner, I remain silent because I am afraid of not being able to control my anger.",
    "I feel I am right in our discussions.",
    "I have nothing to do with what I've been accused of.",
    "I'm not actually the one who's guilty about what I'm accused of.",
    "I'm not the one who's wrong about problems at home.",
    "I wouldn't hesitate to tell my partner about their inadequacy.",
    "When I discuss, I remind my partner of their inadequate issues.",
    "I'm not afraid to tell my partner about their incompetence.",
]

SCALE_LABELS = ["0 – Never", "1 – Rarely", "2 – Sometimes", "3 – Often", "4 – Always"]


def train_and_save_model():
    df = pd.read_csv(CSV_PATH, sep=";")
    x = df.drop("Class", axis=1)
    y = df["Class"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.15, random_state=42
    )
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(x_train, y_train)
    train_acc = clf.score(x_train, y_train)
    test_acc  = clf.score(x_test, y_test)
    print(f"Model trained — Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
    # Try writing to bundle dir first (local dev); fall back to /tmp (Vercel)
    save_path = MODEL_SAVE
    try:
        joblib.dump(clf, MODEL_PATH)
        save_path = MODEL_PATH
    except OSError:
        joblib.dump(clf, MODEL_SAVE)
    print(f"Model saved to {save_path}")
    return clf, test_acc


# Load or train model on startup.
# On Vercel, /tmp is empty on cold start so we always load the bundled pkl.
_load_path = MODEL_SAVE if os.path.exists(MODEL_SAVE) else MODEL_PATH
if os.path.exists(_load_path):
    model = joblib.load(_load_path)
    print(f"Model loaded from {_load_path}")
else:
    model, _ = train_and_save_model()


@app.route("/")
def index():
    return render_template("index.html", questions=QUESTIONS, scale_labels=SCALE_LABELS)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    answers = data.get("answers", [])
    if len(answers) != 54:
        return jsonify({"error": f"Expected 54 answers, got {len(answers)}."}), 400

    try:
        values = [int(a) for a in answers]
        features = np.array(values).reshape(1, -1)
        prediction = int(model.predict(features)[0])
        proba = model.predict_proba(features)[0]
        divorce_prob  = round(float(proba[1]) * 100, 1)
        stable_prob   = round(float(proba[0]) * 100, 1)
        return jsonify({
            "prediction": prediction,
            "label": "High Risk of Divorce" if prediction == 1 else "Marriage Appears Stable",
            "divorce_probability": divorce_prob,
            "stable_probability": stable_prob,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/retrain", methods=["POST"])
def retrain():
    global model
    model, acc = train_and_save_model()
    return jsonify({"message": "Model retrained successfully.", "test_accuracy": round(acc * 100, 2)})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
