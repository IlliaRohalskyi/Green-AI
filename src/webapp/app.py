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

        # Überprüfe, ob die Checkbox für Performance aktiviert ist
        performance_needs = request.form.get("input3")
        if request.form.get("performance_important"):
            performance_needs = "this is not important for me"

        # Überprüfe, ob die Checkbox für Time aktiviert ist
        time = request.form.get("input4")
        if request.form.get("time_important"):
            time = "this is not important for me"

        # Überprüfe, ob die Checkbox für Budget aktiviert ist
        budget = request.form.get("input5")
        if request.form.get("budget_important"):
            budget = "this is not important for me"

        # Überprüfe, ob die Checkbox für Eco Friendliness aktiviert ist
        eco_friendliness = request.form.get("input6")
        if request.form.get("eco_important"):
            eco_friendliness = "this is not important for me"

        # Übergib die Eingaben an main und hole das Ergebnis
        try:
            result = main(task, data, performance_needs, time, budget, eco_friendliness)
            result_data = json.loads(result)

            # Zugriff auf verschachtelte Felder im `architecturer` oder `calculator`
            architecturer_data = result_data.get('architecturer', {})
            weight_reasoning = architecturer_data.get('weight_reasoning', 'Kein Eintrag')
            model_architecture = architecturer_data.get('model_architecture', 'Kein Eintrag')
            training_strategy = architecturer_data.get('training_strategy', 'Kein Eintrag')
            architecture_reasoning = architecturer_data.get('architecture_reasoning', 'Kein Eintrag')
        
        except Exception as e:
            # Fehlerbehandlung
            return render_template("form.html", error=f"Error: {str(e)}")
        
        return render_template("form.html", 
                               result_data=result_data,
                               weight_reasoning=weight_reasoning,
                               model_architecture=model_architecture,
                               training_strategy=training_strategy,
                               architecture_reasoning=architecture_reasoning)
    
    return render_template("form.html")

if __name__ == "__main__":
    app.run(debug=True)
