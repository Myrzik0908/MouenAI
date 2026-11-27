This document outlines the file structure and format for a model repository in C.

### File Header

The file header is used to identify the file and contains the following fields:

|    Field     | Data Type | Number of Bytes |
|--------------|-----------|-----------------|
| Magic String |  char[6]  |        6        |
| File Version | uint32_t  |        4        |

### Model Structure

The model consists of the following fields:

|       Field         |       Data Type       | Number of Bytes |
|---------------------|-----------------------|-----------------|
| Model Configuration |       uint64_t        |        8        |
| Model Weights       |         float         |   (variable)    |
| Model Gradients     |         float         |   (variable)    |
| Optimizer Steps     |       uint64_t        |        8        |
| Momentum for AdamW  |         float         |        4        |
| RMSprop for AdamW   |         float         |        4        |
| Tokenizer           | (See tokenizer files) |   (variable)    |

### Field Descriptions

- **Magic String**: A 6-character string used to identify the file format.
- **File Version**: An unsigned 32-bit integer indicating the version of the file format.
- **Model Configuration**: 
  - `vocab_size`: An unsigned 64-bit integer representing the vocabulary size.
  - `position_size`: An unsigned 64-bit integer representing the position size.
  - `d_model`: An unsigned 64-bit integer representing the dimensionality of the model.
  - `num_heads`: An unsigned 64-bit integer representing the number of attention heads.
  - `num_layers`: An unsigned 64-bit integer representing the number of transformer layers.
  - `ffn_dim`: An unsigned 64-bit integer representing the dimensionality of the feedforward network.
  - `eps`: A float representing the epsilon value for numerical stability.
- **Model Weights**: A sequence of floats representing the model weights.
- **Model Gradients**: A sequence of floats representing the gradients of the model.
- **Optimizer Steps**: An unsigned 64-bit integer representing the number of optimizer steps.
- **Momentum for AdamW**: A float representing the momentum value for AdamW.
- **RMSprop for AdamW**: A float representing the RMSprop value for AdamW.
- **Tokenizer**: A structure representing the tokenizer configuration. The exact structure is defined in the tokenizer files.

### Functions

1. **SaveModel**
   - Saves the full model with all gradients or only the configuration and weights.
   - Parameters:
     - `model`: A pointer to the model to save.
     - `with_gradients`: A boolean indicating whether to save the entire model (1) or just the configuration and weights (0).
     - `filepath`: The path for saving the file.
   - Returns:
     - `int`: 1 for error, 0 for success.

2. **LoadModel**
   - Creates a model and loads its weights from a file.
   - Parameters:
     - `filepath`: The path to the model file.
     - `use_mmap`: An integer indicating whether to use memory mapping (0 for allocate + read, 1 for mmap file).
   - Returns:
     - `Model*`: A pointer to the created model on success, `NULL` on error.