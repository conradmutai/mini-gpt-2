import random
import torch

import torch.optim as optim
import torch.nn as nn

from pathlib import Path

from .model import GPT
from .data import Tokenizer, load_dir, batching, make_batches

# loading the data
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "f1_wiki_pages"  # creates a universal path to the data
corpus = load_dir(DATA_DIR)

torch.manual_seed(0)
random.seed(0)

# building the vocab, tokens and token ids, as well as calculating the vocab_size
tokenizer = Tokenizer()
tokenizer.build_vocab(corpus)
token_ids = tokenizer.encode(corpus)
vocab_size = tokenizer.vocab_size

# initializing model, optimizers and loss
model = GPT(vocab_size, max_seq_len=128, num_heads=8, num_layers=5, d_model=128)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()


def train(epochs):
    # batching
    batch_list = batching(tokenizer, corpus, seq_len=128)

    # splitting the batching into training windows and validation windows
    split_idx = int(len(batch_list) * 0.9)
    train_windows = batch_list[:split_idx]
    val_windows = batch_list[split_idx:]

    batches = make_batches(train_windows, batch_size=32)
    val_batches = make_batches(val_windows, batch_size=32, shuffle=False)  # validation batches

    loss_list = []
    val_loss_list = []

    # print(len(batch_list))  # shows the length of the batch list for test

    for e in range(epochs):  # looping over the epochs
        # shuffle the training data each epoch
        random.shuffle(batches)

        epoch_losses = []

        for inputs, targets in batches:
            # forward pass
            outputs = model(inputs)
            outputs = outputs.view(-1, vocab_size)  # reshapes the outputs and targets to enable loss calculation
            targets = targets.view(-1)

            loss = criterion(outputs, targets)  # calculates the losses

            # backward pass
            optimizer.zero_grad()  # resets the gradient
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        loss_list.append(avg_loss)

        # changes the model to an evaluation stage
        model.eval()
        val_losses = []

        # disables the grad descent on torch
        with torch.no_grad():

            # loops over the validation batches
            for inputs, targets in val_batches:
                outputs = model(inputs)
                outputs = outputs.view(-1, vocab_size)
                targets = targets.view(-1)
                val_losses.append(criterion(outputs, targets).item())

        # switches the model into train mode
        model.train()

        avg_val = sum(val_losses) / len(val_losses)
        val_loss_list.append(avg_val)
        print(f"epoch: {e + 1}, train loss: {round(avg_loss, 4)}, val loss: {round(avg_val, 4)}")

    return loss_list, val_loss_list

# testing the model
if __name__ == '__main__':
    train(epochs=10)
