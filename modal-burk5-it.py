import modal

app = modal.App("burk-5-it")

image = (
   modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .uv_pip_install("torch==2.11.0")
    .uv_pip_install("numpy==2.4.4")
    .uv_pip_install("transformers==5.13.0")
    .uv_pip_install("requests==2.34.2")
)

cache_vol = modal.Volume.from_name("burk-5-it-cache", create_if_missing=True)

# Import required libraries
import json            # For parsing JSON data
import random          # For setting seeds and shuffling data
import requests        # For downloading dataset from URL
import torch           # Main PyTorch library
from torch.utils.data import Dataset, DataLoader  # For dataset handling
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria  # HuggingFace components
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

def build_prompt(instruction, solution=None):
    """
    Creates a chat-formatted prompt with system, user, and assistant messages.

    Args:
        instruction (str): User's instruction/question
        solution (str, optional): Expected response for training

    Returns:
        str: Formatted prompt string
    """
    # Add solution with end token if provided
    wrapped_solution = ""
    if solution:
        wrapped_solution = f"\n{solution}\n<|im_end|>"

    # Build chat format with system, user, and assistant messages
    return f"""<|im_start|>system
You are a helpful assistant.
<|im_end|>
<|im_start|>user
{instruction}
<|im_end|>
<|im_start|>assistant""" + wrapped_solution

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

class EndTokenStoppingCriteria(StoppingCriteria):
    """
    Custom stopping criteria for text generation.
    Stops when a specific end token sequence is generated.

    Args:
        end_tokens (list): Token IDs that signal generation should stop
        device: Device where the model is running
    """
    def __init__(self, end_tokens, device):
        self.end_tokens = torch.tensor(end_tokens).to(device)

    def __call__(self, input_ids, scores):
        """
        Checks if generation should stop for each sequence.

        Args:
            input_ids: Current generated token IDs
            scores: Token probabilities

        Returns:
            tensor: Boolean tensor indicating which sequences should stop
        """
        should_stop = []

        # Check each sequence for end tokens
        for sequence in input_ids:
            if len(sequence) >= len(self.end_tokens):
                # Compare last tokens with end tokens
                last_tokens = sequence[-len(self.end_tokens):]
                should_stop.append(torch.all(last_tokens == self.end_tokens))
            else:
                should_stop.append(False)

        return torch.tensor(should_stop, device=input_ids.device)

class PromptCompletionDataset(Dataset):
    """
    PyTorch Dataset for instruction-completion pairs.
    Handles the conversion of text data into model-ready format.

    Args:
        data (list): List of dictionaries containing instructions and solutions
        tokenizer: Hugging Face tokenizer
    """
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        # Return total number of examples
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns a single training example.

        Args:
            idx (int): Index of the example to fetch

        Returns:
            dict: Contains input_ids, labels, prompt, and expected completion
        """
        # Get example from dataset
        item = self.data[idx]
        # Build full prompt with instruction
        prompt = build_prompt(item["instruction"])
        # Format completion with end token
        completion = f"""{item["solution"]}\n<|im_end|>"""

        # Convert text to token IDs
        encoded_prompt = encode_text(self.tokenizer, prompt)
        encoded_completion = encode_text(self.tokenizer, completion)
        eos_token = [self.tokenizer.eos_token_id]

        # Combine for full input sequence
        input_ids = encoded_prompt + encoded_completion + eos_token
        # Create labels: -100 for prompt (ignored in loss)
        labels = [-100] * len(encoded_prompt) + encoded_completion + eos_token

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
    # Find longest sequence for padding
    max_length = max(len(item["input_ids"]) for item in batch)

    # Pad input sequences
    input_ids = [
        item["input_ids"] +
        [tokenizer.pad_token_id] * (max_length - len(item["input_ids"]))
        for item in batch
    ]
    # Pad label sequences
    labels = [
        item["labels"] +
        [-100] * (max_length - len(item["labels"]))
        for item in batch
    ]
    # Create attention masks
    attention_mask = [
        [1] * len(item["input_ids"]) +
        [0] * (max_length - len(item["input_ids"]))
        for item in batch
    ]
    prompts = [item["prompt"] for item in batch]
    expected_completions = [item["expected_completion"] for item in batch]

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
    text = re.sub(r'\s+', ' ', text)
    return text

def generate_text(model, tokenizer, prompt, max_new_tokens=100):
    """
    Generates text completion for a given prompt.

    Args:
        model: Fine-tuned model
        tokenizer: Associated tokenizer
        prompt (str): Input prompt
        max_new_tokens (int): Maximum number of tokens to generate

    Returns:
        str: Generated completion
    """
    # Encode prompt and move to model's device
    input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Setup end token detection
    end_tokens = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    stopping_criteria = [EndTokenStoppingCriteria(end_tokens, model.device)]

    # Generate completion
    output_ids = model.generate(
        input_ids=input_ids["input_ids"],
        attention_mask=input_ids["attention_mask"],
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        stopping_criteria=stopping_criteria
    )[0]

    # Extract and decode only the generated part
    generated_ids = output_ids[input_ids["input_ids"].shape[1]:]
    generated_text = tokenizer.decode(generated_ids).strip()
    return generated_text

def test_model(model_path, test_input):
    """
    Tests a saved model on a single input.

    Args:
        model_path (str): Path to saved model
        test_input (str): Instruction to test
    """
    # Setup device and load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    # Generate and display prediction
    prompt = build_prompt(test_input)
    generated_text = generate_text(model, tokenizer, prompt)

    print(f"\nInput: {test_input}")
    print(f"Full generated text: {generated_text}")
    print(f"""Cleaned response: {generated_text.replace("<|im_end|>", "").strip()}""")

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
    # Download dataset
    response = requests.get(data_url)
    dataset = []
    # Parse each line as an instruction-solution pair
    for line in response.text.splitlines():
        if line.strip():  # Skip empty lines
            entry = json.loads(line)
            dataset.append({
                "instruction": entry["instruction"],
                "solution": entry["solution"]
            })

    # Split into train and test sets
    random.shuffle(dataset)
    split_index = int(len(dataset) * (1 - test_ratio))
    train_data = dataset[:split_index]
    test_data = dataset[split_index:]

    # Print dataset statistics
    print(f"\nDataset size: {len(dataset)}")
    print(f"Training samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")

    # Create datasets
    train_dataset = PromptCompletionDataset(train_data, tokenizer)
    test_dataset = PromptCompletionDataset(test_data, tokenizer)

    # Create dataloaders
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
    # Fewer epochs for instruction tuning as it's more data-efficient
    num_epochs = 4
    # Standard batch size that works well with most GPU memory
    batch_size = 16
    # Standard learning rate for fine-tuning transformers
    learning_rate = 5e-5

    return num_epochs, batch_size, learning_rate

@app.function(image=image,gpu="A10:4",timeout=15*60*60,volumes={"/vol":cache_vol})
def fine_tune():
    # Set random seed for reproducibility
    set_seed(42)

    # Configure training parameters
    data_url = "https://www.thelmbook.com/data/instruct"
    model_name = "openai-community/gpt2"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    # Get hyperparameters and prepare data
    num_epochs, batch_size, learning_rate = get_hyperparameters()
    train_loader, test_loader = download_and_prepare_data(data_url, tokenizer, batch_size)

    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

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

        # Display epoch metrics
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1} - Average loss: {avg_loss:.4f}")

    # Save the fine-tuned model
    model.save_pretrained("/vol/finetuned_model")
    tokenizer.save_pretrained("/vol/finetuned_model")

    # Test the model
    print("\nTesting finetuned model:")
    test_input = "Who is the President of the United States?"
    test_model("/vol/finetuned_model", test_input)

@app.local_entrypoint()
def main():
    print(fine_tune.spawn().get())
