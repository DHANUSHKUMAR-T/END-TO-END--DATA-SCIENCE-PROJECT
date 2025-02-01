from flask import Flask, request, render_template
import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns
import os

app = Flask(__name__)

model = joblib.load('C:\\Users\\Dhanush\\OneDrive\\Desktop\\churn_prediction_project\\heart_disease_prediction\\saved_model\\heart_disease_model.pkl')

if not os.path.exists("static"):
    os.makedirs("static")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
   
    data = [float(request.form[key]) for key in request.form]
    features = list(request.form.keys())

    data = np.array(data).reshape(1, -1)

    prediction = model.predict(data)

    #bar chart for visualization
    plt.figure(figsize=(10, 5))
    sns.barplot(x=features, y=data[0], palette="coolwarm")
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Features")
    plt.ylabel("Values")
    plt.title("Input Data Visualization")

    graph_path = "static/heart_disease_graph.png"
    plt.savefig(graph_path)
    plt.close()
    
    #ECG Signal Graph
    t = np.linspace(0, 1, 500)
    ecg_signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 10 * t)

    plt.figure(figsize=(8, 3))
    plt.plot(t, ecg_signal, color='red', lw=2)
    plt.title("Simulated ECG Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    ecg_path = "static/ecg_wave.png"
    plt.savefig(ecg_path)
    plt.close()
    
    #Pie Chart for Heart Disease Risk Factors
    risk_factors = ["Age", "Cholesterol", "Blood Pressure", "Exercise", "Smoking"]
    risk_values = [30, 25, 20, 15, 10]  # Example values

    plt.figure(figsize=(6, 6))
    plt.pie(risk_values, labels=risk_factors, autopct='%1.1f%%', colors=['red', 'orange', 'yellow', 'blue', 'green'])
    plt.title("Heart Disease Risk Factor Analysis")

    risk_path = "static/heart_risk_pie.png"
    plt.savefig(risk_path)
    plt.close()

    #Cholesterol vs Age
    ages = np.random.randint(30, 80, 50)
    cholesterol = np.random.randint(150, 300, 50)

    plt.figure(figsize=(8, 5))
    plt.scatter(ages, cholesterol, color='blue', alpha=0.6)
    plt.xlabel("Age (years)")
    plt.ylabel("Cholesterol Level (mg/dL)")
    plt.title("Cholesterol Levels vs Age")
    plt.grid(True)

    cholesterol_path = "static/cholesterol_vs_age.png"
    plt.savefig(cholesterol_path)
    plt.close()
    
    return render_template('result.html', prediction=prediction[0], graph_path="static/heart_disease_graph.png", ecg_path=ecg_path, risk_path=risk_path, cholesterol_path=cholesterol_path)

if __name__ == "__main__":
    app.run(debug=False,host='0.0.0.0')
