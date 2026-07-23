import json

import torch
import torch.nn.functional as F

from pathlib import Path

from .model import GPT
from .data import Tokenizer

# file paths for results to use in generate.py
RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "best_model.pt"
VOCAB_PATH = Path(__file__).resolve().parent.parent / "results" / "vocab.json"

# output list
results = []

# initializing the tokenizer
tokenizer = Tokenizer()

# loading the word_to_id from the JSON that was created in training
with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    tokenizer.word_to_id = json.load(f)

tokenizer.id_to_word = {int(v): k for k, v in tokenizer.word_to_id.items()}

vocab_size = tokenizer.vocab_size

model = GPT(vocab_size, max_seq_len=128, num_heads=8, num_layers=5, d_model=128)
model.load_state_dict(torch.load(RESULTS_PATH))  # loading the best val loss from the training
model.eval()  # setting the model into evaluation state


def generate(text_input, max_new_tokens=100, max_seq_len=128):
    id_list = tokenizer.encode(text_input)

    for _ in range(max_new_tokens):
        context = id_list[-max_seq_len:]  # gathers contex from the id list
        inputs = torch.tensor(context).view(1, len(context))  # makes the context of a tensor and transforms the shape

        outputs = model(inputs)
        prediction = outputs[:, -1, :]

        # Option 1: Greedy Algorithm
        # probs = torch.argmax(prediction).item()
        # id_list.append(probs)

        # Option 2: Top K
        prediction = prediction / 1.2
        values, indices = torch.topk(prediction, k=15, dim=1, largest=True, sorted=True)
        probs = F.softmax(values, dim=1)
        next_id = torch.multinomial(probs, num_samples=1).item()
        real_id = indices[:, next_id].item()

        id_list.append(real_id)

    return tokenizer.decode(id_list)


if __name__ == '__main__':
    text_input_test = input("Enter a prompt: ")
    print(generate(text_input_test))