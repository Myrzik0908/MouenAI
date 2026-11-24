import ctypes, re
from collections import Counter
from collections.abc import Iterable

class GreedyTokenizer:
    def __init__(
            self,
            start_vocab: list[str] | None = None,
            max_len_vocab: int=5000,
            min_frequency: int = 2,
            max_token_len: int = 16
    ) -> None:
        """
        A greedy tokenizer takes the longest sequence.
        The unknown token is already built into the tokenizer!
        :param start_vocab:
        :param max_len_vocab:
        :param min_frequency:
        :param max_token_len:
        """
        # settings
        self.base_vocab: set[str] = set(start_vocab or [])
        self.max_len_vocab: int = max_len_vocab
        self.min_frequency: int = min_frequency
        self.max_token_len: int = max_token_len
        self.token_counts: Counter[str] = Counter()
        # runtime
        self.vocab = sorted(set(self.base_vocab), key=lambda x: (-len(x.encode('utf-8')), x))
        self.max_char_len: int = 0
        self.stoi: dict[str, int] = {}
        self.to_index()

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
            pattern: str = r'\w+|\s+|[^\w\s]'
    ) -> None:
        text = (dataset_corpus,) if isinstance(dataset_corpus, str) else dataset_corpus
        counts = self.token_counts
        min_frequency = self.min_frequency
        max_len = self.max_token_len
        compiled_pattern = re.compile(pattern)
        for example in text:
            fragments = compiled_pattern.findall(example)
            for chunk in fragments:
                n = len(chunk)
                if n == 1:
                    counts[chunk] += 1
                else:
                    for s in range(n):
                        for e in range(s + 1, min(n, s + max_len) + 1):
                            counts[chunk[s:e]] += 1
        candidates = [
            (tok, count) for tok, count in counts.items()
            if (count >= min_frequency or len(tok) == 1) and tok not in self.base_vocab
        ]
        candidates.sort(
            key=lambda x: float('inf') if len(x[0]) == 1 else x[1] * (len(x[0]) ** 0.5),
            reverse=True
        )
        limit = max(0, self.max_len_vocab - len(self.base_vocab))
        top_tokens = [tok for tok, _ in candidates[:limit]]
        self.vocab = sorted(
            self.base_vocab.union(top_tokens),
            key=lambda x: (-len(x.encode('utf-8')), x)
        )
        self.to_index()