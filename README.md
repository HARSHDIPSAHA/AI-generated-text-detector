# AI-Generated Text Detector

This repository contains code for fine-tuning a BERT model to detect AI-generated text. The model can classify text as either 'student' generated or 'AI' generated.

## Setup Instructions

### Prerequisites

- Python 3.12
- Jupyter Notebook
- CUDA-enabled GPU (optional but recommended for faster training)
- Required Python packages (listed in `requirements.txt`)

### Installation

1. **Clone the repository:**

    ```sh
    git clone https://github.com/HARSHDIPSAHA/AI-generated-text-detector.git
    cd AI-generated-text-detector
    ```

2. **Create a virtual environment:**

    ```sh
    python -m venv myenv
    source myenv/bin/activate   # On Windows, use `myenv\Scripts\activate`
    ```

3. **Install the required packages:**

    ```sh
    pip install -r requirements.txt
    ```

### Running the Notebook

1. **Launch Jupyter Notebook:**

    ```sh
    jupyter notebook
    ```

2. **Open `berttttt.ipynb` in Jupyter Notebook.**

3. **Run all cells to train the model and make predictions.**

### Usage

#### Training the Model

- The notebook includes code to load data, tokenize text, and train a BERT model for sequence classification.
- The training process is configured to run for 5 epochs with a learning rate of `2e-5`.

#### Making Predictions

- After training, the model and tokenizer are saved for future use.
- The notebook demonstrates how to load the trained model and tokenizer and use them to classify new text inputs.

### Example Predictions

```python
text = "Blockchain is revolutionizing education, providing personalized learning experiences to students all over the world."
predicted_label = predict_text_category(text, model, tokenizer)
print(f"The predicted label for the given text is: {predicted_label}")
