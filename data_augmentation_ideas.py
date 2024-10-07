import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.model_selection import train_test_split
from datasets import Dataset
import pandas as pd
import nltk
from nltk.corpus import wordnet

# Download necessary NLTK data
nltk.download('wordnet')
# Synonym replacement function
def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms)

def synonym_replacement(sentence, n=1):
    words = sentence.split()
    new_sentence = words.copy()
    random_word_list = list(set(words))
    random.shuffle(random_word_list)

    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(synonyms)
            new_sentence = [synonym if word == random_word else word for word in new_sentence]
            num_replaced += 1
        if num_replaced >= n:
            break

    return ' '.join(new_sentence)

# Augment the training data
def augment_data(df, n_augments=1):
    augmented_texts = []
    labels = []
    for _, row in df.iterrows():
        text = row['content']
        label = row['label']
        augmented_texts.append(text)  # Original text

        # Generate augmented versions of the text
        for _ in range(n_augments):
            augmented_text = synonym_replacement(text)
            augmented_texts.append(augmented_text)
            labels.append(label)

    augmented_df = pd.DataFrame({'content': augmented_texts, 'label': labels})
    return augmented_df

# Apply augmentation
augmented_train_df = augment_data(train_df, n_augments=2)  # Augment each text twice

# Convert the augmented dataframe to Dataset
augmented_train_dataset = Dataset.from_pandas(augmented_train_df)
augmented_train_dataset = augmented_train_dataset.map(tokenize_function, batched=True)

from transformers import BertConfig, BertForSequenceClassification
# Define dropout in model config
config = BertConfig.from_pretrained(model_name, 
                                    hidden_dropout_prob=0.3, 
                                    attention_probs_dropout_prob=0.3)

model = BertForSequenceClassification.from_pretrained(model_name, config=config).to(device)

# Define training arguments with weight decay and gradient clipping
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,  # Regularization
    max_grad_norm=1.0,  # Gradient clipping
    fp16=torch.cuda.is_available(),
)

# Data Augmentation (as described before)
augmented_train_df = augment_data(train_df, n_augments=2)
augmented_train_dataset = Dataset.from_pandas(augmented_train_df)
augmented_train_dataset = augmented_train_dataset.map(tokenize_function, batched=True)

# Define Trainer with early stopping
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=augmented_train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],  # Early stopping
)

# Train model
trainer.train()

# Save model and tokenizer
save_model_and_tokenizer(model, tokenizer, model_path, tokenizer_path)
