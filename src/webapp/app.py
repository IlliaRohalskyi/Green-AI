from flask import Flask, render_template, request
from src.main import main  # Import the main function from main.py
import os
import json

# Absolute path to the templates and static directory
template_dir = os.path.abspath('src/webapp/templates')
static_dir = os.path.abspath('src/webapp/static')

app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Get form inputs
        task = request.form.get("input1")
        data = request.form.get("input2")

        # Check if the checkbox for Performance is checked
        performance_needs = request.form.get("input3")
        if request.form.get("performance_important"):
            performance_needs = "this is not important for me"

        # Check if the checkbox for Time is checked
        time = request.form.get("input4")
        if request.form.get("time_important"):
            time = "this is not important for me"

        # Check if the checkbox for Budget is checked
        budget = request.form.get("input5")
        if request.form.get("budget_important"):
            budget = "this is not important for me"

        # Check if the checkbox for Eco Friendliness is checked
        eco_friendliness = request.form.get("input6")
        if request.form.get("eco_important"):
            eco_friendliness = "this is not important for me"

        # Pass inputs to main and get the result
        try:
            result = main(task, data, performance_needs, time, budget, eco_friendliness)
            result_data = json.loads(result)

            # Access nested fields from `architecturer` or `calculator`
            architecturer_data = result_data.get('architecturer', {})
            weight_reasoning = architecturer_data.get('weight_reasoning', 'No entry')
            model_architecture = architecturer_data.get('model_architecture', 'No entry')
            training_strategy = architecturer_data.get('training_strategy', 'No entry')
            architecture_reasoning = architecturer_data.get('architecture_reasoning', 'No entry')
        
        except Exception as e:
            # Error handling
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
