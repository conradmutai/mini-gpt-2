import re
import torch
import random
from pathlib import Path

from collections import Counter


class Tokenizer:
    # regex pattern — alternatives ordered most-specific-first
    PATTERN = r'''(?x)
     (?:\d{1,2}:)?\d{1,2}:\d{2}(?:\.\d+)?    # race/lap times
    |(?:[A-Z]\.)+                            # abbreviations
    |\w+(?:-\w+)*                            # words, optional hyphens
    |\$?\d+(?:\.\d+)?%?                      # currency, numbers, percentages
    |[][.,;"'?():_`-]                        # standalone punctuation
    '''

    UNK = "<UNK>"

    def __init__(self):
        self.regex = re.compile(self.PATTERN, re.VERBOSE)
        self.word_to_id = {}
        self.id_to_word = {}

    @property
    def vocab_size(self):
        return len(self.word_to_id)

    def tokenize(self, text):
        return re.findall(self.regex, text)  # turns text into tokens based off the pattern

    def build_vocab(self, corpus, min_freq=3):
        tokens = self.tokenize(corpus)
        counts = Counter(tokens)

        unique = sorted(w for w, c in counts.items() if c >= min_freq)

        self.word_to_id = {self.UNK: 0}
        self.id_to_word = {0: self.UNK}

        # converts the tokens into ids and vice versa
        for i, word in enumerate(unique, start=1):
            self.word_to_id[word] = i
            self.id_to_word[i] = word

    # encodes the text by turning tokens into token ids
    def encode(self, corpus):
        tokens = self.tokenize(corpus)
        return [self.word_to_id.get(token, 0) for token in tokens]

    # turns token ids back into tokens
    def decode(self, ids):
        text = " ".join(self.id_to_word.get(i, self.UNK) for i in ids)
        text = re.sub(r'\s+([.,;:!?)\]])', r'\1', text)  # no space before closing punctuation
        text = re.sub(r"(['\"(\[])\s+", r'\1', text)  # no space after opening punctuation/quotes
        return text


# for loading files in directory
def load_dir(dir_path):
    return_str = []  # list used to store all words from the files loaded
    directory = Path(dir_path)

    for item in directory.iterdir():
        if item.is_file():
            with open(item, "r", encoding="utf-8") as f:
                return_str.append(f.read())  # appending contents of file to the str list

    return '\n\n'.join(return_str)  # making it into a singular string with spacing in between


def batching(tokenizer, corpus, seq_len, stride=None):
    if stride is None:
        stride = seq_len  # default: non-overlapping, unchanged behaviour

    token_ids = tokenizer.encode(corpus)  # encoding the text into token ids
    num_windows = (len(token_ids) - 1) // seq_len  # creating the appropriate amount of windows
    batch_list = []

    for i in range(num_windows):
        start = i * seq_len  # the starting point of the batch
        chunk = token_ids[start: start + seq_len + 1]  # creates a chunk with the token ids

        input_seq = chunk[0: seq_len]  # the initial sequence
        target_seq = chunk[1: seq_len + 1]  # the sequence to follow

        batch_list.append((input_seq, target_seq))  # added to batch list

    return batch_list


def make_batches(batch_list, batch_size, shuffle=True):
    if shuffle:
        random.shuffle(batch_list)  # shuffles the batch list to make better training data

    batches = []  # list to store batches
    num_batches = len(batch_list) // batch_size

    for b in range(num_batches):
        chunk = batch_list[b * batch_size: (b + 1) * batch_size]  # creates the chunk

        inputs = torch.tensor([pair[0] for pair in chunk], dtype=torch.long)  # makes the inputs a tensor and same for targets
        targets = torch.tensor([pair[1] for pair in chunk], dtype=torch.long)

        batches.append((inputs, targets))  # appends this to batches

    return batches
