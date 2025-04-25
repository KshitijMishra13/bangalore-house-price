from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load model and columns
model = pickle.load(open("Ridge.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=["POST"])
def predict():
    try:
        location = request.form['location'].lower()
        bhk = int(request.form['bhk'])
        bath = int(request.form['bathrooms'])
        balcony = int(request.form['balcony'])
        sqft = float(request.form['area'])

        x = np.zeros(len(columns))
        x[0] = sqft
        x[1] = bath
        x[2] = balcony
        x[3] = bhk

        if location in columns:
            loc_index = columns.index(location)
            x[loc_index] = 1

        prediction = model.predict([x])[0]
        price = round(prediction, 2)
        return render_template("index.html", prediction_text=f"Estimated Price: ₹ {price:,.2f}")

    except Exception as e:
        return render_template("index.html", prediction_text="Invalid Input")

if __name__ == "__main__":
    app.run(debug=True)
