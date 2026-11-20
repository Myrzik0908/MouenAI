import ctypes
import weakref
import numpy as np
from numpy.ctypeslib import ndpointer
from enum import IntEnum
from .loader import get_lib

_SIZE_T = np.dtype(f"u{ctypes.sizeof(ctypes.c_size_t)}")

class TokenizerType(IntEnum):
    TOK_GREEDY = 0

class TokenizerStruct(ctypes.Structure):
    pass

class Tokenizer:
    def __init__(self, tokenizer: TokenizerType, filepath: str=None):
        self.lib = get_lib()
        self._setup()
        ptr = self.lib.tok_create(
            int(tokenizer),
            filepath.encode("utf-8") if filepath is not None else None
        )
        if not ptr:
            raise RuntimeError(f"tok_create failed: {tokenizer.name} / {filepath}")
        self._ptr = ptr
        self._buf = None
        self._n = 0

        self._finalizer = weakref.finalize(
            self,
            self.lib.tok_destroy,
            ptr
        )

    def _setup(self):
        lib = self.lib

        # Tokenizer* tok_create(TokenizerType type, const char* filepath);
        lib.tok_create.argtypes = [ctypes.c_int, ctypes.c_char_p]
        lib.tok_create.restype = ctypes.POINTER(TokenizerStruct)

        # void tok_printf(const Tokenizer* self);
        lib.tok_printf.argtypes = [ctypes.POINTER(TokenizerStruct)]
        lib.tok_printf.restype = None

        # void tok_destroy(Tokenizer* self);
        lib.tok_destroy.argtypes = [ctypes.POINTER(TokenizerStruct)]
        lib.tok_destroy.restype = None

        # size_t tok_encode(const Tokenizer* self, const char* sequense, size_t* buffer, const size_t len_buffer);
        lib.tok_encode.argtypes = [
            ctypes.POINTER(TokenizerStruct),
            ctypes.c_char_p,
            ndpointer(dtype=_SIZE_T, ndim=1, flags="C_CONTIGUOUS"),
            ctypes.c_size_t
        ]
        lib.tok_encode.restype = ctypes.c_size_t

        # void tok_tokenprintf(Tokenizer* self, const size_t token);
        lib.tok_tokenprintf.argtypes = [ctypes.POINTER(TokenizerStruct), ctypes.c_size_t]
        lib.tok_tokenprintf.restype = None

        # size_t tok_num_tokens(const Tokenizer* self);
        lib.tok_num_tokens.argtypes = [ctypes.POINTER(TokenizerStruct)]
        lib.tok_num_tokens.restype = ctypes.c_size_t

        # int tok_import_tokenizer_from_bin(Tokenizer* self, const char* filepath);
        lib.tok_import_tokenizer_from_bin.argtypes = [ctypes.POINTER(TokenizerStruct), ctypes.c_char_p]
        lib.tok_import_tokenizer_from_bin.restype = ctypes.c_int

    def close(self) -> None:
        self._finalizer()
        self._ptr = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __repr__(self):
        state = "closed" if self._ptr is None else f"ptr={self._ptr}"
        return f"<Tokenizer {state}>"

    def printf(self) -> None:
        if self._ptr is None:
            raise RuntimeError("Tokenizer already closed")
        self.lib.tok_printf(self._ptr)

    def load_vocab(self, filepath: str) -> None:
        if self._ptr is None:
            raise RuntimeError("Tokenizer already closed")
        if not filepath:
            raise ValueError("filepath must be non-empty")

        result = self.lib.tok_import_tokenizer_from_bin(
            self._ptr,
            filepath.encode("utf-8"),
        )
        if result != 0:
            raise RuntimeError(
                f"tok_import_tokenizer_from_bin failed: result={result}, file={filepath!r}"
            )

    def num_tokens(self) -> int:
        if self._ptr is None:
            raise RuntimeError("Tokenizer already closed")
        return self.lib.tok_num_tokens(self._ptr)

    def token_print(self, token: int) -> None:
        if self._ptr is None:
            raise RuntimeError("Tokenizer already closed")
        if token is None or token < 0:
            raise ValueError(f"Invalid token value: {token}")
        self.lib.tok_tokenprintf(self._ptr, token)

    def encode(self, text: str, max_tokens: int=1024) -> np.ndarray:
        '''Encode text. Returns a VIEW into an internal buffer.
        The view is valid ONLY until the next encode() call.
        Copy it if you need to keep it: result.copy().'''
        if self._ptr is None:
            raise RuntimeError("Tokenizer already closed")
        if not text:
            raise ValueError("text must be non-empty")

        if self._buf is None or self._buf.size < max_tokens:
            self._buf = np.empty(max_tokens, dtype=_SIZE_T)
        buf = self._buf

        n = self.lib.tok_encode(
            self._ptr,
            text.encode("utf-8"),
            buf,
            buf.size,
        )

        if n == 0:
            raise RuntimeError("The tokenizer received 0 tokens")
        self._n = n
        return buf[:n]

    def append(self, token: int) -> np.ndarray:
        if self._ptr is None:
            raise RuntimeError("Tokenizer already closed")
        if self._buf is None:
            raise RuntimeError("call encode() first")

        if self._n >= self._buf.size:
            new_buf = np.empty(self._buf.size * 2, dtype=_SIZE_T)
            new_buf[:self._n] = self._buf[:self._n]
            self._buf = new_buf
        self._buf[self._n] = token
        self._n += 1
        return self._buf[:self._n]