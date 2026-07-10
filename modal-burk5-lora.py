import modal

app = modal.App("burk-5-lora")

image = (
   modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .uv_pip_install("torch==2.11.0")
    .uv_pip_install("numpy==2.4.4")
    .uv_pip_install("transformers==5.13.0")
    .uv_pip_install("requests==2.34.2")
    .uv_pip_install("peft==0.19.1")
)

cache_vol = modal.Volume.from_name("burk-5-lora-cache", create_if_missing=True)

# Import required libraries
import json            # For parsing JSON data
import random          # For setting seeds and shuffling data
import gzip            # For decompressing dataset
import requests        # For downloading dataset from URL
import torch           # Main PyTorch library
from peft import get_peft_model, LoraConfig, TaskType  # For efficient finetuning using LoRA
from torch.utils.data import Dataset, DataLoader  # For dataset handling
from transformers import AutoTokenizer, AutoModelForCausalLM  # Hugging Face model components
from torch.optim import AdamW    # Optimizer for training
from tqdm import tqdm   # Progress bar utilities
import re               # For text normalization

def set_seed(seed):
    """
    Sets random seeds for reproducibility across different libraries.

    Args:
        seed (int): Seed value for random number generation
    """
    # Set Python's built-in random seed
    random.seed(seed)
    # Set PyTorch's CPU random seed
    torch.manual_seed(seed)
    # Set seed for all available GPUs
    torch.cuda.manual_seed_all(seed)
    # Request cuDNN to use deterministic algorithms
    torch.backends.cudnn.deterministic = True
    # Disable cuDNN's auto-tuner for consistent behavior
    torch.backends.cudnn.benchmark = False

def build_prompt(text):
    """
    Creates a standardized prompt for emotion classification.

    Args:
        text (str): Input text to classify

    Returns:
        str: Formatted prompt for the model
    """
    # Format the input text into a consistent prompt structure
    return f"Predict the emotion for the following text: {text}\nEmotion:"

def encode_text(tokenizer, text, return_tensor=False):
    """
    Encodes text using the provided tokenizer.

    Args:
        tokenizer: Hugging Face tokenizer
        text (str): Text to encode
        return_tensor (bool): Whether to return PyTorch tensor

    Returns:
        List or tensor of token IDs
    """
    # If tensor output is requested, encode with PyTorch tensors
    if return_tensor:
        return tokenizer.encode(
            text, add_special_tokens=False, return_tensors="pt"
        )
    # Otherwise return list of token IDs
    else:
        return tokenizer.encode(text, add_special_tokens=False)

def decode_text(tokenizer, token_ids):
    """
    Decodes token IDs back to text.

    Args:
        tokenizer: Hugging Face tokenizer
        token_ids: List or tensor of token IDs

    Returns:
        str: Decoded text
    """
    # Convert token IDs back to text, skipping special tokens
    return tokenizer.decode(token_ids, skip_special_tokens=True)

class PromptCompletionDataset(Dataset):
    """
    PyTorch Dataset for prompt-completion pairs.
    Handles the conversion of text data into model-ready format.

    Args:
        data (list): List of dictionaries containing prompts and completions
        tokenizer: Hugging Face tokenizer
    """
    def __init__(self, data, tokenizer):
        # Store the raw data and tokenizer for later use
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        # Return the total number of examples in the dataset
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns a single training example.

        Args:
            idx (int): Index of the example to fetch

        Returns:
            dict: Contains input_ids, labels, prompt, and expected completion
        """
        # Get the specific example from our dataset
        item = self.data[idx]
        prompt = item["prompt"]
        completion = item["completion"]

        # Convert text to token IDs for both prompt and completion
        encoded_prompt = encode_text(self.tokenizer, prompt)
        encoded_completion = encode_text(self.tokenizer, completion)
        # Get the end-of-sequence token ID
        eos_token = self.tokenizer.eos_token_id

        # Combine prompt and completion tokens with EOS token
        input_ids = encoded_prompt + encoded_completion + [eos_token]
        # Create labels: -100 for prompt (ignored in loss), completion tokens for learning
        labels = [-100] * len(encoded_prompt) + encoded_completion + [eos_token]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "prompt": prompt,
            "expected_completion": completion
        }

def collate_fn(batch):
    """
    Collates batch of examples into training-ready format.
    Handles padding and conversion to tensors.

    Args:
        batch: List of examples from Dataset

    Returns:
        tuple: (input_ids, attention_mask, labels, prompts, expected_completions)
    """
    # Initialize tokenizer
    model_name = "openai-community/gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Find the longest sequence in the batch for padding
    max_length = max(len(item["input_ids"]) for item in batch)

    # Pad input sequences to max_length with pad token
    input_ids = [
        item["input_ids"] +
        [tokenizer.pad_token_id] * (max_length - len(item["input_ids"]))
        for item in batch
    ]

    # Pad label sequences with -100 (ignored in loss calculation)
    labels = [
        item["labels"] +
        [-100] * (max_length - len(item["labels"]))
        for item in batch
    ]

    # Create attention masks: 1 for real tokens, 0 for padding
    attention_mask = [
        [1] * len(item["input_ids"]) +
        [0] * (max_length - len(item["input_ids"]))
        for item in batch
    ]

    # Keep original prompts and completions for evaluation
    prompts = [item["prompt"] for item in batch]
    expected_completions = [item["expected_completion"] for item in batch]

    # Convert everything to PyTorch tensors except text
    return (
        torch.tensor(input_ids),
        torch.tensor(attention_mask),
        torch.tensor(labels),
        prompts,
        expected_completions
    )

def normalize_text(text):
    """
    Normalizes text for consistent comparison.

    Args:
        text (str): Input text

    Returns:
        str: Normalized text
    """
    # Remove leading/trailing whitespace and convert to lowercase
    text = text.strip().lower()
    # Replace multiple whitespace characters with single space
    text = re.sub(r"\s+", ' ', text)
    return text

def calculate_accuracy(model, tokenizer, loader):
    """
    Calculates prediction accuracy on a dataset.

    Args:
        model: Finetuned model
        tokenizer: Associated tokenizer
        loader: DataLoader containing evaluation examples

    Returns:
        float: Accuracy score
    """
    # Set model to evaluation mode
    model.eval()
    # Initialize counters for accuracy calculation
    correct = 0
    total = 0

    # Disable gradient computation for efficiency
    with torch.no_grad():
        for input_ids, attention_mask, labels, prompts, expected_completions in loader:
            for prompt, expected_completion in zip(prompts, expected_completions):
                # Generate model's prediction
                generated_text = generate_text(model, tokenizer, prompt)
                # Compare normalized versions of prediction and target
                if normalize_text(generated_text) == normalize_text(expected_completion):
                    correct += 1
                total += 1

    # Calculate accuracy, handling empty dataset case
    accuracy = correct / total if total > 0 else 0
    # Reset model to training mode
    model.train()
    return accuracy

def generate_text(model, tokenizer, prompt, max_new_tokens=50):
    """
    Generates text completion for a given prompt.

    Args:
        model: Finetuned model
        tokenizer: Associated tokenizer
        prompt (str): Input prompt
        max_new_tokens (int): Maximum number of tokens to generate

    Returns:
        str: Generated completion
    """
    # Encode prompt and move to model's device
    input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate completion using model's generate method
    output_ids = model.generate(
        input_ids=input_ids["input_ids"],
        attention_mask=input_ids["attention_mask"],
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )[0]

    # Extract and decode only the generated part (excluding prompt)
    generated_text = decode_text(tokenizer, output_ids[input_ids["input_ids"].shape[1]:])
    return generated_text.strip()

def test_model(model_path, test_input):
    """
    Tests a saved model on a single input.

    Args:
        model_path (str): Path to saved model
        test_input (str): Text to classify
    """
    # Determine device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load saved model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Configure padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    # Generate and display prediction
    prompt = build_prompt(test_input)
    generated_text = generate_text(model, tokenizer, prompt)

    print(f"Input: {test_input}")
    print(f"Generated emotion: {generated_text}")

def download_and_prepare_data(data_url, tokenizer, batch_size, test_ratio=0.1):
    """
    Downloads and prepares dataset for training.

    Args:
        data_url (str): URL of the dataset
        tokenizer: Tokenizer for text processing
        batch_size (int): Batch size for DataLoader
        test_ratio (float): Proportion of data for testing

    Returns:
        tuple: (train_loader, test_loader)
    """
    # Download and decompress dataset
    response = requests.get(data_url)
    content = gzip.decompress(response.content).decode()

    # Process each example into prompt-completion pairs
    dataset = []
    for entry in map(json.loads, content.splitlines()):
        dataset.append({
            "prompt": build_prompt(entry['text']),
            "completion": entry["label"].strip()
        })

    # Split into train and test sets
    random.shuffle(dataset)
    split_index = int(len(dataset) * (1 - test_ratio))
    train_data = dataset[:split_index]
    test_data = dataset[split_index:]

    # Create datasets
    train_dataset = PromptCompletionDataset(train_data, tokenizer)
    test_dataset = PromptCompletionDataset(test_data, tokenizer)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    return train_loader, test_loader

def get_hyperparameters():
    """
    Returns training hyperparameters.

    Returns:
        tuple: (num_epochs, batch_size, learning_rate)
    """
    # Train for more epochs with LoRA as it's more efficient
    num_epochs = 18
    # Batch size
    batch_size = 16
    # Standard learning rate for finetuning transformers
    learning_rate = 5e-5

    return num_epochs, batch_size, learning_rate

@app.function(image=image,gpu="A10:4",timeout=15*60*60,volumes={"/vol":cache_vol})
def fine_tune():
    # Set random seeds for reproducibility
    set_seed(42)

    # Configure basic training parameters
    data_url = "https://www.thelmbook.com/data/emotions"
    model_name = "openai-community/gpt2"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Configure LoRA parameters
    peft_config = LoraConfig(
        task_type = TaskType.CAUSAL_LM,  # Set task type for causal language modeling
        inference_mode = False,          # Enable training mode
        r = 16,                          # Rank of LoRA update matrices
        lora_alpha = 32                  # LoRA scaling factor
    )

    # Load model and apply LoRA configuration
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model = get_peft_model(model, peft_config)

    # Get hyperparameters and prepare data
    num_epochs, batch_size, learning_rate = get_hyperparameters()
    train_loader, test_loader = download_and_prepare_data(data_url, tokenizer, batch_size)

    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for input_ids, attention_mask, labels, _, _ in progress_bar:
            # Move batch to device
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            progress_bar.set_postfix({"Loss": total_loss / num_batches})

        # Calculate and display epoch metrics
        avg_loss = total_loss / num_batches
        test_acc = calculate_accuracy(model, tokenizer, test_loader)
        print(f"Epoch {epoch+1} - Average loss: {avg_loss:.4f}, Test accuracy: {test_acc:.4f}")

    # Calculate final model performance
    train_acc = calculate_accuracy(model, tokenizer, train_loader)
    print(f"Training accuracy: {train_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    # Save the LoRA-tuned model and tokenizer
    model.save_pretrained("/vol/finetuned_model")
    tokenizer.save_pretrained("/vol/finetuned_model")

    # Test the finetuned model with a sample input
    test_input = "I'm so happy to be able to finetune an LLM!"
    test_model("/vol/finetuned_model", test_input)

@app.local_entrypoint()
def main():
    print(fine_tune.remote())
