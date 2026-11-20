import ctypes
from collections import Counter
from collections.abc import Iterable

class GreedyTokenizer:
    def __init__(self, start_vocab: list[str] | None = None, max_len_vocab: int=5000) -> None:
        """
        A greedy tokenizer takes the longest sequence.
        The unknown token is already built into the tokenizer!
        :param start_vocab:
        :param max_len_vocab:
        """
        self.vocab = sorted(set(start_vocab or []), key=lambda x: (-len(x.encode('utf-8')), x))
        self.max_char_len: int = 0
        self.stoi: dict[str, int] = {}
        self.to_index()
        self.max_len_vocab = max_len_vocab

    def show_vocab(self) -> None:
        """
        Outputs each token on a new line
        :return:
        """
        print(f"MAX CHAR LEN: {self.max_char_len}")
        print("0: system unknown token")
        for i, token in enumerate(self.vocab):
            print(f"{i + 1}: '{token}'")

    def to_index(self) -> None:
        """
        Indexes a dictionary for fast encoding
        :return:
        """
        self.max_char_len = max((len(t) for t in self.vocab), default=0)
        self.stoi = {s: i for i, s in enumerate(self.vocab, start=1)}

    def save_bytes(self, path: str) -> None:
        """
        Save vocab to the format binary file:
        - num_tokens
        - array leights of tokens
        - array tokens
        :return:
        """
        vocab_bytes = [token.encode('utf-8') for token in self.vocab]
        with open(path, "wb") as f:
            f.write(ctypes.c_uint64(len(vocab_bytes)))
            for token_bytes in vocab_bytes:
                f.write(ctypes.c_uint64(len(token_bytes)))
            for token_bytes in vocab_bytes:
                f.write(token_bytes)
            f.write(ctypes.c_char(b'\0')) # End-of-line character

    def encode(self, text: str="") -> list[int]:
        """
        Converts text into an array of indices
        :param text:
        :return:
        """
        if not text:
            return []
        encoded = []
        i = 0
        n = len(text)
        while i < n:
            found = False
            for length in range(min(self.max_char_len, n - i), 0, -1):
                candidate = text[i:i + length]
                if candidate in self.stoi:
                    encoded.append(self.stoi[candidate])
                    i += length
                    found = True
                    break

            if not found:
                encoded.append(0)  # ID 0 – unknown
                i += 1
        return encoded

    def decode(self, data: list[int] | None = None) -> str:
        """
        Converts an array of indices into text
        :param data:
        :return:
        """
        if not data:
            return ''

        vocab_len = len(self.vocab)
        return ''.join(self.vocab[i - 1] if 0 <= i - 1< vocab_len else '' for i in data)

    def train(
            self,
            dataset_corpus: str | Iterable[str],
            vocab_size: int | None = None,
            min_frequency: int = 2,
            max_token_len: int = 16
    ) -> None:
        pass