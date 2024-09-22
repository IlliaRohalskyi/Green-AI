from flask import Flask, render_template, request
from src.main import main  # Import the main function from main.py
import os
import json

# Absolute path to the templates and static directories
template_dir = os.path.abspath('src/webapp/templates')
static_dir = os.path.abspath('src/webapp/static')

app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Capture the new inputs
        max_time = request.form.get("max_time")
        max_cost = request.form.get("max_cost")
        max_co2 = request.form.get("max_co2")
        
        # If the user didn't input anything, use the defaults
        max_time = float(max_time) if max_time else 1e19
        max_cost = float(max_cost) if max_cost else None
        max_co2 = float(max_co2) if max_co2 else None

        # Capture other inputs and process them as before
        task = request.form.get("input1")
        data = request.form.get("input2")
        performance_needs = request.form.get("input3") if not request.form.get("performance_important") else "this is not important for me"
        time = request.form.get("input4") if not request.form.get("time_important") else "this is not important for me"
        budget = request.form.get("input5") if not request.form.get("budget_important") else "this is not important for me"
        eco_friendliness = request.form.get("input6") if not request.form.get("eco_important") else "this is not important for me"

        # Pass the inputs to the main function including max_time, max_cost, and max_co2
        try:
            result = main(task, data, performance_needs, time, budget, eco_friendliness, max_time=max_time, max_cost=max_cost, max_co2=max_co2)
            result_data = json.loads(result)
        except Exception as e:
            return render_template("form.html", error=f"Error: {str(e)}")
        
        return render_template("form.html", result_data=result_data)
    
    return render_template("form.html")

if __name__ == "__main__":
    app.run(debug=True)
