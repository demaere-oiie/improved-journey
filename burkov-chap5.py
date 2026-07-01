import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

#set_seed(42)
data_url = "https://www.thelmbook.com/data/emotions"
model_name = "openai-community/gpt2"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

def get_hyperparameters():
    return (2, 16, 5e-5)

num_epochs, batch_size, learning_rate = get_hyperparameters()

train_loader, test_loader = download_and_prepare_data(
    data_url, tokenizer, batch_size
)
