# Handwritten Digit Identifier

This project is a Streamlit app that identifies handwritten digits using a Convolutional Neural Network (CNN). The app takes an image of a handwritten digit and predicts the digit using a pre-trained model.

## Project Structure

- `cnn.pth`: The pre-trained CNN model file.
- `identify_number.ipynb`: Jupyter notebook for training and evaluating the model.
- `model.py`: Python script defining the CNN model architecture.
- `requirements.txt`: List of dependencies required to run the project.
- `streamlit_app.py`: Streamlit app script for the handwritten digit identifier.

## Getting Started

To run the Streamlit app on your local machine after cloning the repo, follow these steps:

1. **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2. **Run the Streamlit app:**
    ```bash
    streamlit run streamlit_app.py
    ```

## TODO

- [ ] Improve the accuracy of the CNN model.
- [ ] Correct and organize the import statements in the scripts.
- [ ] Rename variables and functions to more meaningful names for better code readability.


Feel free to open issues or contribute to this project by submitting a pull request.
