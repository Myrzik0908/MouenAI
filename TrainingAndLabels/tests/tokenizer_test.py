from DatasetTools import GreedyTokenizer
from MouenAPI import Tokenizer, TokenizerType
import os

# EXAMPLE of your dataset
line = "hello, world!🙋‍♀️Have a nice day.🥰"

# example
py_tok = GreedyTokenizer()
py_tok.train(line)
py_tok.save_bytes("test.bin")
py_encode = py_tok.encode(line)
print("Python:")
print(f"encode: {py_encode}, decode: \n{py_tok.decode(py_encode)}")

c_tok = Tokenizer(TokenizerType.TOK_GREEDY, "test.bin")
print("C backend:")
c_encode = c_tok.encode(line)
print(f"encode: {c_encode}, decode: ")
for x in c_encode:
    c_tok.token_print(int(x))

# cleaning
os.remove("test.bin")

# inspections
assert py_tok.decode(py_encode) == line, "The decrypted text does not match the original!"
assert py_encode == list(c_encode), "Differences between tokenizers in Python and C!"