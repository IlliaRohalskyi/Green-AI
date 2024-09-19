from flask import Flask, render_template, request
from src.main import main  # Importiere die main Funktion aus main.py
import os
import json

# Absoluter Pfad zum Templates-Verzeichnis
template_dir = os.path.abspath('src/webapp/templates')
static_dir = os.path.abspath('src/webapp/static')

app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Hole die Formulareingaben
        task = request.form.get("input1")
        data = request.form.get("input2")
        performance_needs = request.form.get("input3")
        time = request.form.get("input4")
        budget = request.form.get("input5")
        eco_friendliness = request.form.get("input6")

        # Übergib die Eingaben an main und hole das Ergebnis
        result = main(task, data, performance_needs, time, budget, eco_friendliness)

        # Die Ausgabe der main.py ist im JSON-Format. Um sie im HTML darzustellen, geben wir sie direkt als Text zurück
        result_data = json.loads(result)
        return render_template("form.html", result_data=result_data)
    
    return render_template("form.html")

if __name__ == "__main__":
    app.run(debug=True)
