from flask import Flask, render_template, request
from src.main import main  
import os
import json

template_dir = os.path.abspath('src/webapp/templates')
static_dir = os.path.abspath('src/webapp/static')

app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)

@app.route("/", methods=["GET", "POST"])
def home():
    """
    The main view function for handling both GET and POST requests to the home route.
    
    If the request method is GET, it renders the form for user input.
    If the request method is POST, it processes the form inputs, 
    calls the `main` function to get the GPU recommendations, and then renders the results.
    
    Input:
        - GET: None
        - POST: User input fields including task, data, performance needs, time, budget, eco-friendliness,
                and optional fields for maximum time, cost, and CO2. 
    
    Output:
        - Renders 'form.html' with either the input form (GET request) or the results (POST request).
    """
    if request.method == "POST":
        task = request.form.get("input1")
        data = request.form.get("input2")
        performance_needs = request.form.get("input3")
        time = request.form.get("input4")
        budget = request.form.get("input5")
        eco_friendliness = request.form.get("input6")
        
        max_time = (float(request.form.get("input_max_time", 1e19))*24*60*60) if request.form.get("input_max_time") else None
        max_cost = float(request.form.get("input_max_cost", None)) if request.form.get("input_max_cost") else None
        max_co2 = float(request.form.get("input_max_co2", None)) if request.form.get("input_max_co2") else None

        if request.form.get("performance_important"):
            performance_needs = "This is not important for me"
        if request.form.get("time_important"):
            time = "This is not important for me"
        if request.form.get("budget_important"):
            budget = "This is not important for me"
        if request.form.get("eco_important"):
            eco_friendliness = "This is not important for me"

        print(f"Task: {task}, Data: {data}, Performance Needs: {performance_needs}, Time: {time}, Budget: {budget}, Eco Friendliness: {eco_friendliness}, Max Time: {max_time}, Max Cost: {max_cost}, Max CO2: {max_co2}")
    
        try:
            result_json, weight_reasoning, model_architecture, training_strategy, architecture_reasoning, cost_weight, time_weight, eco_weight = main(
                task, data, performance_needs, time, budget, eco_friendliness,
                max_time=max_time, max_cost=max_cost, max_co2=max_co2
            )
            result_data = json.loads(result_json)
        except Exception as e:
            return render_template("form.html", error=f"Error: {str(e)}")

        return render_template(
            "form.html",
            result_data=result_data,
            weight_reasoning=weight_reasoning,
            model_architecture=model_architecture,
            training_strategy=training_strategy,
            architecture_reasoning=architecture_reasoning,

            cost_weight = cost_weight,
            time_weight = time_weight,
            eco_weight = eco_weight

        )

    return render_template("form.html")

if __name__ == "__main__":
    """
    Start the Flask web application in debug mode.
    
    The app will run locally and listen for requests on the default port (5000).
    """
    app.run(debug=True)
