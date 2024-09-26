# GreenAI

GreenAI is an intelligent system that optimizes AI model training by recommending hardware configurations that balance performance, cost, and environmental impact. It provides users with tailored suggestions for GPU selection based on their specific project requirements and constraints.

## Features

- **Task-Specific Recommendations**: Input your AI task, dataset details, and performance needs to receive customized hardware suggestions.
- **Flexible Constraint Management**: Set priorities for eco-friendliness, time efficiency, and cost, including optional maximum limits.
- **Comprehensive GPU Evaluation**: Each GPU is scored based on CO2 emissions, time efficiency, and cost-effectiveness.
- **Intuitive User Interface**: Easy-to-use input form and results display with star ratings and tooltips for detailed insights.
- **Intelligent Decision Making**: Utilizes a language model to analyze inputs and provide reasoned architecture and training strategy recommendations.

## Installation and Setup

Follow these steps to set up GreenAI on your local machine:

1. Ensure you have **Python 3.11.9** installed. You can download it from [python.org](https://www.python.org/downloads/).

2. Clone the repository:
    ```bash
    git clone https://github.com/IlliaRohalskyi/Green-AI.git
    cd Green-AI
    ```

3. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Download and install Ollama:
    Visit the [Ollama website](https://ollama.com/) and follow the installation instructions for your operating system.

5. Install Phi3 Mini
   ```bash
   ollama pull phi3:mini
   ```

6. Run the application:
    ```bash
    python /src/webapp/app.py
    ```

This will start the GreenAI web application. You can access it by opening a web browser and navigating to the address (typically `http://localhost:5000`).

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
- Text classification (e.g., sentiment analysis)
- Computer vision tasks (e.g., object detection for autonomous vehicles)
- Image processing (e.g., stock photo categorization)

## Future Improvements

- Multi-GPU support for parallel processing recommendations
- Expanded GPU database for more hardware options
- Chatbot integration for guided user input

## Contributing

We welcome contributions to improve GreenAI. To contribute:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
5. Push to the branch (`git push origin feature/AmazingFeature`)
6. Open a Pull Request

Please ensure your code adheres to the project's coding standards and include tests for new features.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or support, please open an issue on this GitHub repository.
