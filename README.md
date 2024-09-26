# GreenAI

GreenAI is an intelligent system that optimizes AI model training by recommending hardware configurations that balance performance, cost, and environmental impact. It provides users with tailored suggestions for GPU selection based on their specific project requirements and constraints.

## Features

- **Task-Specific Recommendations**: Input your AI task, dataset details, and performance needs to receive customized hardware suggestions.
- **Flexible Constraint Management**: Set priorities for eco-friendliness, time efficiency, and cost, including optional maximum limits.
- **Comprehensive GPU Evaluation**: Each GPU is scored based on CO2 emissions, time efficiency, and cost-effectiveness.
- **Intuitive User Interface**: Easy-to-use input form and results display with star ratings and tooltips for detailed insights.
- **Intelligent Decision Making**: Utilizes a language model to analyze inputs and provide reasoned architecture and training strategy recommendations.

## How It Works

1. **Input**: Users provide task description, dataset information, performance needs, and constraints.
2. **Processing**: The system uses a graph-based decision-making engine to:
   - Assign priorities based on user preferences
   - Recommend model architecture and training strategy
   - Calculate and rank GPU configurations
3. **Output**: Displays a ranked list of GPU options with detailed metrics and explanations.

## System Architecture

- **Frontend**: Handles user input and result display
- **Business Logic**: Validates inputs and manages data flow
- **AI Component**: Processes data and generates recommendations using:
  - Functions for calculations and ranking
  - GPU and environmental impact data
  - Language Model (Phi3 Mini) for natural language understanding

## Getting Started

1. Define your AI task and dataset details
2. Specify performance needs, time constraints, budget, and eco-friendliness priorities
3. (Optional) Set maximum limits for time, cost, and CO2 emissions
4. Submit your request
5. Review the recommended GPU configurations and their ratings

## Use Cases

GreenAI is suitable for various AI projects, including:
- Natural language processing tasks (e.g., sentiment analysis classification)
- Computer vision tasks (e.g., object detection for autonomous vehicles)


## Contributing

We welcome contributions to improve GreenAI.

## License

Apache License 2.0