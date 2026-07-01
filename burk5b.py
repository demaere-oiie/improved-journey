# Import required libraries
import torch           # Main PyTorch library
from torch.utils.data import DataLoader  # For dataset handling
from torch.optim import AdamW    # Optimizer for training
from transformers import AutoTokenizer, AutoModelForSequenceClassification  # Hugging Face components
from tqdm import tqdm   # Progress bar utilities
import json             # For parsing JSON data
import requests         # For downloading dataset from URL
import gzip             # For decompressing dataset
import random           # For setting seeds and shuffling data

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

def load_and_split_dataset(url, test_ratio=0.1):
    """
    Downloads and splits dataset into train and test sets.

    Args:
        url (str): URL of the dataset
        test_ratio (float): Proportion of data for testing

    Returns:
        tuple: (train_dataset, test_dataset)
    """
    # Download and decompress dataset
    response = requests.get(url)
    content = gzip.decompress(response.content).decode()

    # Parse JSON lines into list of examples
    dataset = [json.loads(line) for line in content.splitlines()]

    # Shuffle and split dataset
    random.shuffle(dataset)
    split_index = int(len(dataset) * (1 - test_ratio))

    return dataset[:split_index], dataset[split_index:]

def load_model_and_tokenizer(model_name, device, label_to_id, id_to_label, unique_labels):
    """
    Loads and configures the model and tokenizer for sequence classification.

    Args:
        model_name (str): Name of pre-trained model
        device: Device to load model on
        label_to_id (dict): Mapping from label strings to IDs
        id_to_label (dict): Mapping from IDs to label strings
        unique_labels (list): List of all possible labels

    Returns:
        tuple: (model, tokenizer)
    """
    # Initialize model with correct number of output classes
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(unique_labels)
    )

    # Configure padding and label mappings
    model.config.pad_token_id = model.config.eos_token_id
    model.config.id2label = id_to_label
    model.config.label2id = label_to_id

    # Initialize and configure tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    return (model.to(device), tokenizer)

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

class TextClassificationDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for text classification.
    Converts text and labels into model-ready format.

    Args:
        data (list): List of dictionaries containing text and labels
        tokenizer: Hugging Face tokenizer
        label_to_id (dict): Mapping from label strings to IDs
    """
    def __init__(self, data, tokenizer, label_to_id):
        self.data = data
        self.tokenizer = tokenizer
        self.label_to_id = label_to_id

    def __len__(self):
        # Return total number of examples
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns a single training example.

        Args:
            idx (int): Index of the example to fetch

        Returns:
            dict: Contains input_ids and labels
        """
        # Get example from dataset
        item = self.data[idx]
        # Convert text to token IDs
        input_ids = encode_text(self.tokenizer, item["text"])
        # Convert label string to ID
        labels = self.label_to_id[item["label"]]

        return {
            "input_ids": input_ids,
            "labels": labels
        }

def collate_fn(batch):
    """
    Collates batch of examples into training-ready format.
    Handles padding and conversion to tensors.

    Args:
        batch: List of examples from Dataset

    Returns:
        dict: Contains input_ids, labels, and attention_mask tensors
    """
    # Find longest sequence for padding
    max_length = max(len(item["input_ids"]) for item in batch)

    # Pad input sequences with zeros
    input_ids = [
        item["input_ids"] +
        [0] * (max_length - len(item["input_ids"]))
        for item in batch
    ]

    # Create attention masks (1 for tokens, 0 for padding)
    attention_mask = [
        [1] * len(item["input_ids"]) +
        [0] * (max_length - len(item["input_ids"]))
        for item in batch
    ]

    # Collect labels
    labels = [item["labels"] for item in batch]

    # Convert everything to tensors
    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(attention_mask)
    }

def generate_label(model, tokenizer, text):
    """
    Generates label prediction for input text.

    Args:
        model: Fine-tuned model
        tokenizer: Associated tokenizer
        text (str): Input text to classify

    Returns:
        str: Predicted label
    """
    # Encode text and move to model's device
    input_ids = encode_text(
        tokenizer,
        text,
        return_tensor=True
    ).to(model.device)

    # Get model predictions
    outputs = model(input_ids)
    logits = outputs.logits[0]
    # Get class with highest probability
    predicted_class = logits.argmax().item()
    # Convert class ID to label string
    return model.config.id2label[predicted_class]

def calculate_accuracy(model, dataloader):
    """
    Calculates prediction accuracy on a dataset.

    Args:
        model: Fine-tuned model
        dataloader: DataLoader containing evaluation examples

    Returns:
        float: Accuracy score
    """
    # Set model to evaluation mode
    model.eval()
    correct = 0
    total = 0

    # Disable gradient computation for efficiency
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            labels = batch["labels"].to(model.device)

            # Get model predictions
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = outputs.logits.argmax(dim=-1)

            # Update accuracy counters
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    # Calculate accuracy
    accuracy = correct / total
    # Reset model to training mode
    model.train()
    return accuracy

def create_label_mappings(train_dataset):
    """
    Creates mappings between label strings and IDs.

    Args:
        train_dataset: List of training examples

    Returns:
        tuple: (label_to_id, id_to_label, unique_labels)
    """
    # Get sorted list of unique labels
    unique_labels = sorted(set(item["label"] for item in train_dataset))
    # Create mappings between labels and IDs
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    id_to_label = {i: label for label, i in label_to_id.items()}
    return label_to_id, id_to_label, unique_labels

def test_model(model_path, test_input):
    """
    Tests a saved model on a single input.

    Args:
        model_path (str): Path to saved model
        test_input (str): Text to classify
    """
    # Setup device and load model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Generate and display prediction
    emotion = generate_label(model, tokenizer, test_input)
    print(f"Input: {test_input}")
    print(f"Predicted emotion: {emotion}")

def download_and_prepare_data(data_url, tokenizer, batch_size):
    """
    Downloads and prepares dataset for training.

    Args:
        data_url (str): URL of the dataset
        tokenizer: Tokenizer for text processing
        batch_size (int): Batch size for DataLoader

    Returns:
        tuple: (train_dataloader, test_dataloader, label_to_id, id_to_label, unique_labels)
    """
    # Load and split dataset
    train_dataset, test_dataset = load_and_split_dataset(data_url)

    # Create label mappings
    label_to_id, id_to_label, unique_labels = create_label_mappings(train_dataset)

    # Create datasets
    train_data = TextClassificationDataset(
        train_dataset,
        tokenizer,
        label_to_id
    )
    test_data = TextClassificationDataset(
        test_dataset,
        tokenizer,
        label_to_id
    )

    # Create dataloaders
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    return train_dataloader, test_dataloader, label_to_id, id_to_label, unique_labels

def get_hyperparameters():
    """
    Returns training hyperparameters.

    Returns:
        tuple: (num_epochs, batch_size, learning_rate)
    """
    # Train for fewer epochs as sequence classification converges faster
    num_epochs=8
    # Standard batch size that works well with most GPU memory
    batch_size=16
    # Standard learning rate for fine-tuning transformers
    learning_rate=5e-5
    return num_epochs, batch_size, learning_rate

# Main training script
if __name__ == "__main__":
    # Set random seed for reproducibility
    seed = 42
    set_seed(seed)

    # Configure training parameters
    data_url = "https://www.thelmbook.com/data/emotions"
    model_name = "openai-community/gpt2"

    # Get hyperparameters
    num_epochs, batch_size, learning_rate = get_hyperparameters()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Prepare data and get label mappings
    train_loader, test_loader, label_to_id, id_to_label, unique_labels = download_and_prepare_data(
        data_url,
        tokenizer,
        batch_size
    )

    # Initialize model for sequence classification
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(unique_labels)
    ).to(device)

    # Configure model's label handling
    model.config.pad_token_id = model.config.eos_token_id
    model.config.id2label = id_to_label
    model.config.label2id = label_to_id

    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch in progress_bar:
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            # Backward pass and optimization
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Update metrics
            total_loss += loss.item()
            num_batches += 1

            progress_bar.set_postfix({"Loss": total_loss / num_batches})

        # Display epoch metrics
        avg_loss = total_loss / num_batches
        test_acc = calculate_accuracy(model, test_loader)
        print(f"Average loss: {avg_loss:.4f}, Test accuracy: {test_acc:.4f}")

    # Save the fine-tuned model
    model.save_pretrained("./finetuned_model")
    tokenizer.save_pretrained("./finetuned_model")

    # Test the model
    test_input = "I'm so happy to be able to finetune an LLM!"
    test_model("./finetuned_model", test_input)
