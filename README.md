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

### Using Git LFS

If you don't want to run Jupyter Notebooks, you can use Git LFS to download the pre-trained model and tokenizer:

1. **Install Git LFS:**

    ```sh
    git lfs install
    ```

2. **Clone the repository with Git LFS:**

    ```sh
    git lfs clone https://github.com/HARSHDIPSAHA/AI-generated-text-detector.git
    cd AI-generated-text-detector
    ```

3. **Copy the `bert_finetuned_model` and `bert_tokenizer` directories to your local machine.**

4. **Create a new Jupyter Notebook and load the model:**

    ```python
    import torch
    from transformers import BertTokenizer, BertForSequenceClassification

    def load_model_and_tokenizer(model_path, tokenizer_path):
        model = BertForSequenceClassification.from_pretrained(model_path)
        tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        return model, tokenizer

    model_path = './bert_finetuned_model'
    tokenizer_path = './bert_tokenizer'
    model, tokenizer = load_model_and_tokenizer(model_path, tokenizer_path)

    labels = {0: "student", 1: "ai"}

    def predict_text_category(dialogue, model, tokenizer):
        inputs = tokenizer(dialogue, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
        return labels[predicted_class]

    text = "Blockchain is revolutionizing education, providing personalized learning experiences to students all over the world."
    predicted_label = predict_text_category(text, model, tokenizer)
    print(f"The predicted label for the given text is: {predicted_label}")
    ```

### Example Predictions

```python
text = "Blockchain is revolutionizing education, providing personalized learning experiences to students all over the world."
predicted_label = predict_text_category(text, model, tokenizer)
print(f"The predicted label for the given text is: {predicted_label}")
