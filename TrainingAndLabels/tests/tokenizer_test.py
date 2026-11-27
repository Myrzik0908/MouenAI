from DatasetTools import GreedyTokenizer
from MouenAPI import Tokenizer, TokenizerType
import os

# EXAMPLE of your dataset
line = "Hello, world!🙋‍♀️Have a nice day.🥰"

# example
print("=" * 30)
print("Python:")
py_tok = GreedyTokenizer()
py_tok.train(line)
py_tok.save_bytes("test.bin")
py_encode = py_tok.encode(line)
print(f"encode: {py_encode}, decode: \n{py_tok.decode(py_encode)}")

# c backend
print("=" * 30)
print("C backend:")
c_tok = Tokenizer(TokenizerType.TOK_GREEDY, "test.bin")
c_encode = c_tok.encode(line)
print(f"encode: {c_encode}, decode: ")
for x in c_encode:
    c_tok.token_print(int(x))
print()

# save checkpoint
py_tok.save_checkpoint("point.json.gz")

# load a checkpoint into a new object
print("=" * 30)
print("Load checkpoint:")
py_tok_new = GreedyTokenizer.load_checkpoint("point.json.gz")
py_new_encode = py_tok_new.encode(line)
print("Python:")
print(f"encode: {py_new_encode}, decode: \n{py_tok_new.decode(py_new_encode)}")

# cleaning
os.remove("test.bin")
os.remove("point.json.gz")

# inspections
assert py_tok.decode(py_encode) == line, "The decrypted text does not match the original!"
assert py_tok_new.decode(py_new_encode) == line, "After loading a checkpoint, the decrypted text does not match the original!"
assert py_tok_new.vocab == py_tok.vocab, "Parameters do not match after loading!"
assert py_tok_new.token_counts == py_tok.token_counts, "After loading a checkpoint, the counter does not match!"
assert py_encode == list(c_encode), "Differences between tokenizers in Python and C!"