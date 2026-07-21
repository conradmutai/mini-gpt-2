import random

import torch.optim as optim
import torch.nn as nn

from pathlib import Path

from .model import GPT
from .data import Tokenizer, load_dir, batching, make_batches

# loading the data
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "f1_wiki_pages"  # creates a universal path to the data
corpus = load_dir(DATA_DIR)

# building the vocab, tokens and token ids, as well as calculating the vocab_size
tokenizer = Tokenizer()
tokenizer.build_vocab(corpus)
token_ids = tokenizer.encode(corpus)
vocab_size = tokenizer.vocab_size

# initializing model, optimizers and loss
model = GPT(vocab_size, max_seq_len=128, num_heads=8, num_layers=10, d_model=256)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()


def train(epochs):
    # batching
    batch_list = batching(tokenizer, corpus, seq_len=128)
    batches = make_batches(batch_list, batch_size=32)

    loss_list = []

    for e in range(epochs):  # looping over the epochs
        # shuffle the training data each epoch
        random.shuffle(batches)

        epoch_losses = []

        for inputs, targets in batches:
            # forward pass
            outputs = model(inputs)
            outputs = outputs.view(-1, vocab_size)  # reshapes the outputs and targets to enable loss calculation
            targets = targets.view(-1)

            loss = criterion(outputs, targets) # calculates the losses

            # adding to list for plotting
            loss_list.append(loss)

            # backward pass
            optimizer.zero_grad()  # resets the gradient
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        loss_list.append(avg_loss)
        print(f"epoch: {e + 1}, avg loss: {round(avg_loss, 4)}")

    return loss_list

