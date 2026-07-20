import re
import torch
import random
from pathlib import Path

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
        return re.findall(self.regex, text)

    def build_vocab(self, corpus):
        tokens = self.tokenize(corpus)
        unique = sorted(set(tokens))

        self.word_to_id = {self.UNK: 0}
        self.id_to_word = {0: self.UNK}

        for i, word in enumerate(unique, start=1):
            self.word_to_id[word] = i
            self.id_to_word[i] = word

    def encode(self, text):
        tokens = self.tokenize(text)
        return [self.word_to_id.get(token, 0) for token in tokens]

    def decode(self, ids):
        return " ".join(self.id_to_word.get(i, self.UNK) for i in ids)


# for loading files in directory
def load_dir(dir_path):
    return_str = []  # list used to store all words from the files loaded
    directory = Path(dir_path)

    for item in directory.iterdir():
        if item.is_file():
            with open(item, "r", encoding="utf-8") as f:
                return_str.append(f.read())  # appending contents of file to the str list

    return '\n\n'.join(return_str)  # making it into a singular string with spacing in between


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


# TESTING PATTERN
# if __name__ == '__main__':
#     pattern = r'''(?x)
#      (?:\d{1,2}:)?\d{1,2}:\d{2}(?:\.\d+)?    # race/lap times
#     |(?:[A-Z]\.)+                            # abbreviations
#     |\w+(?:-\w+)*                            # hyphen words
#     |\$?\d+(?:\.\d+)?%?                      # currency, percentages
#     |[][.,;"'?():_`-]                        # separate tokens
#     '''
#
#     test = re.findall(pattern, "That U.S.A. poster-print costs $12.40...")
#     print(test)
