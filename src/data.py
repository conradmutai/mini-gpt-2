import re


import re


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
