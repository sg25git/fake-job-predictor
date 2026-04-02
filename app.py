from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    # Get all inputs
    title = request.form['title']
    location = request.form['location']
    description = request.form['description']
    company = request.form.get('company', '')
    requirements = request.form.get('requirements', '')
    benefits = request.form.get('benefits', '')
    employment_type = request.form.get('employment_type', '')
    experience = request.form.get('experience', '')

    # Combine text
    text = " ".join([
        title, location, description,
        company, requirements, benefits,
        employment_type, experience
    ])

    # Predict
    vector = vectorizer.transform([text])
    prediction = model.predict(vector)[0]

    # Confidence
    try:
        prob = model.predict_proba(vector)[0][prediction]
        confidence = round(prob * 100, 2)
    except:
        confidence = None

    result = "Fake Job ❌" if prediction == 1 else "Real Job ✅"

    # 🔥 IMPORTANT: SEND BACK ALL INPUT VALUES
    return render_template(
        "index.html",
        prediction=result,
        confidence=confidence,
        title=title,
        location=location,
        description=description,
        company=company,
        requirements=requirements,
        benefits=benefits,
        employment_type=employment_type,
        experience=experience
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)