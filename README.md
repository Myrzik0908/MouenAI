# MouenAI - transformer engine in pure C

---

A transformer engine in C99 with OpenMP, cache-friendly data layout, and a Python layer via the C-API

---

### Peculiarities:
 - 1. Pure C: The engine core (forward/backward passes, tensors, optimizers) is written in C, free from heavy dependencies.
 - 2. Transformer from Scratch: Implements the Transformer architecture, including Multi-Head Attention, LayerNorm, and residual connections.
 - 3. SiLU (Swish) Activation: Uses $x \cdot \text{sigmoid}(x)$ instead of ReLU or GELU.
 - 4. KV-Cache & Prefix Matching: Caches context during inference, recomputing only the tokens that have changed.
 - 5. AdamW Optimizer: A built-in optimizer supporting weight decay, gradient clipping, and momentum accumulation.
 - 6. Zero-Copy Python API: Bindings via `ctypes` pass NumPy arrays to C without memory copying. A Python-side RAII wrapper ensures no memory leaks occur on the C side.

### Technology stack:
 - 1. Core: C (C11/C17/C23)
 - 2. Build: CMake (+ OpenMP for parallel computing)
 - 3. API: Python 3.11+, ctypes, numpy
 - 4. Building on Windows: LLVM/Clang only (MSVC does not support the new `enum : int` syntax or the project's specific requirements).

### Installation and assembly:

1. Clone the repository:
```bash
git clone https://github.com/yourname/MouenAI.git
cd MouenAI
```

2. Build the library using CMake:
*   **Linux/macOS:**
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```
*   **Windows (LLVM/Clang):**
```cmd
cmake -B build -G "Ninja" -DCMAKE_C_COMPILER=clang -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

3. Install Python dependencies:
```bash
pip install numpy
```

### ⚡ Quick Start (Python API)

An example of creating a model and training it on XOR:

```python
from MouenAPI import Tokenizer, TokenizerType, Model, WorkspaceType, engine_size_t
import numpy as np

# Tokenizer example
tok = Tokenizer(TokenizerType.TOK_GREEDY)

# Example of creating a model and workspaces
mod = Model()
mod.create(2, 4096, 256, 8, 4, 1024, 1e-3)
mod.init_model(0.02)
mod.init_workspace(WorkspaceType.ENGINE_TRAINING, 512)
mod.init_train()

# About the model
mod.show_details()

# Simple educational examples(XOR)
xor_x = [[0, 0], [1, 0], [0, 1], [1, 1]]
xor_y = [[0], [1], [1], [0]]

# Train loop
for epoch in range(35):
    glob_loss = 0
    for i, y in enumerate(xor_y):
        x = xor_x[i]
        # A zero-copy array (dtype: size_t) is passed to C.
        arr = np.array([x[0], x[1], y[0]], dtype=engine_size_t)
        label = np.array([0, 0, 1], dtype=np.uint8)  # last token

        loss = mod.train_step(arr, arr, label)
        glob_loss += loss

    # AdamW optimizer step (lr, beta1, beta2, eps, weight_decay, grad_clip)
    norm = mod.optimizer_step(0.001, 0.9, 0.999, 1e-8, 0.01, 1.0)
    print(f"epoch {epoch}: loss={glob_loss / 4:.4f}, norm={norm:.4f}")
```