#pragma once
#include <stdio.h>
#include <stdint.h>
#include "tokenizer.h"

#define MOUEN_MAGIC "MOUEN"
#define MOUEN_FILE_VERSION 0

#ifdef TRAINING
#define MOUEN_TRAINING 1
#else
#define MOUEN_TRAINING 0
#endif

typedef enum : int {
	ENGINE_INFERENCE = 0,
	ENGINE_TRAINING = 1,
} WorkspaceType;

// Stores the model configuration
typedef struct {
	size_t vocab_size;      // user
	size_t position_size;   // user
	size_t d_model;         // user
	size_t num_heads;       // user
	size_t head_dim;        // automatically
	size_t head_size;       // automatically
	size_t num_layers;      // user
	size_t ffn_dim;         // user
	size_t layer_size;		// automatically
	float eps;				// user
} Config;

// Stores model weights and gradients
typedef struct {
	// arrays
	float* data;

	// encoding
	float* embedding;
	float* positional;
	// layer
	float* W_q;
	float* W_k;
	float* W_v;
	float* W_o;
	float* W_ffn1;
	float* W_ffn2;
	float* gamma1;
	float* beta1;
	float* gamma2;
	float* beta2;
	// head
	float* lm_head;

#if MOUEN_TRAINING
	float* grad;
	// encoding
	float* grad_embedding;
	float* grad_positional;
	// layer
	float* grad_W_q;
	float* grad_W_k;
	float* grad_W_v;
	float* grad_W_o;
	float* grad_W_ffn1;
	float* grad_W_ffn2;
	float* grad_gamma1;
	float* grad_beta1;
	float* grad_gamma2;
	float* grad_beta2;
	// head
	float* grad_lm_head;
#endif

	void* mmap_ptr;       // pointer
	size_t size_model;    // float elements
	size_t size_mmap;     // bytes (for munmap)
	uint8_t is_mmap;      // 0 = malloc, 1 = mmap
} Weights;

// Stores forward and backward pass data.
typedef struct {
	// arrays
	float* forward;
	// forward pointers
	float* encoding;
	float* layernorm1;
	float* q;
	float* k;
	float* v;
	float* att;
	float* o;
	float* attention;
	float* layernorm2;
	float* ff_out1;
	float* ff_out2;
	float* logits;
#if MOUEN_TRAINING
	// training forward pointers
	float* mean1;
	float* inv_std1;
	float* x_hat1;
	float* concat;
	float* att_residual;
	float* mean2;
	float* inv_std2;
	float* x_hat2;
	float* ff_pre_act;
	float* ff_residual;

	float* backward;
	// backward pointers
	float* dlogits;
	float* dY;
	float* dffn1;
	float* dlayernorm;
	float* dconcat;
	float* dO_head;
	float* datt;
	float* dQ;
	float* dV;
	float* dK;

	// pointers for local arrays from LayerNorm
	float* thread_gamma;
	float* thread_beta;
#endif 

	size_t* updated;
	WorkspaceType mode;
	size_t context;
	size_t layer_inference_size;
	size_t chunk_size;
#if MOUEN_TRAINING
	size_t total_backward;
	size_t batchs; // It is needed for the iterative accumulation of a batch.
	size_t num_threads; // Required for multithreading in layernorm
#endif
} Workspace;


#if MOUEN_TRAINING
// AdamW
typedef struct {
	float* momentum;
	float* rmsprop;
	size_t step;
	float learning_rate, beta1, beta2;
} AdamW;
#endif


// It unites all the structures.
typedef struct {
	Config* config;
	Weights* weights;
	Workspace* workspace;
	Tokenizer* tokenizer;
#if MOUEN_TRAINING
	AdamW* adamw;
#endif
} Model;

/*
 * Returns the mode in which the engine was built
 * return(int): 1 is Training, 0 is only Inference
 */
int _build_type(void);

/*
 * The function creates a model object and accepts the following parameters:
 * vocab_size – vocabulary size
 * position_size – maximum number of positions
 * d_model – layer dimensionality
 * num_heads – number of attention heads
 * num_layers – number of attention layers
 * ffn_dim – feed-forward layer size
 * eps – hyperparameter for LayerNorm
 */
Model* CreateModel(
	const size_t vocab_size,
	const size_t position_size,
	const size_t d_model, 
	const size_t num_heads,
	const size_t num_layers,
	const size_t ffn_dim,
	const float eps
);

/*
 * Initializes the model
 * model – pointer to the model structure
 * stddev - standard Deviation
 */
void InitModel(
	Model* model,
	const float stddev
);

/*
 * Initializes memory arrays for model inference.
 * model – pointer to the model structure
 * type - in what mode will the model be launched
 * chunk_size - maximum number of tokens processed in a single pass during inference mode
 * return: int -> 1 error, 0 succes
 */
int InitWorkspace(
	Model* model,
	WorkspaceType type,
	size_t chunk_size
);

/*
 * Executes one forward and sampling step of the model and returns the predicted token
 * model – pointer to the model structure
 * inputs - pointer to the array of input values
 * inputs_size - size of the input sequence
 * temperature - temperature for variable token prediction
 * Returns: predicted token index on success, SIZE_MAX on error.
 */
size_t InferenceStep(
	Model* model,
	const size_t* inputs,
	const size_t inputs_size,
	const float temperature
);

#if MOUEN_TRAINING
/*
 * Initializes memory arrays for model training.
 * model – pointer to the model structure
 * return: int -> 1 error, 0 succes
 */
int InitTrain(
	Model* model
);

/*
 * Performs a single gradient accumulation step.
 * model – pointer to the model structure
 * inputs - pointer to the array of input values
 * targets - pointer to the array of target values
 * labels - pointer to the labels for the training value mask (0 - skip gradient for this value, 1 - calculate gradient for this value)
 * inputs_size - size of the input sequence
 */
float TrainStep(
	Model* model,
	const size_t* inputs,
	const size_t* target,
	const uint8_t* labels,
	const size_t inputs_size
);

/*
 * Performs a single step of the AdamW optimizer, updating the weights and resetting their gradients to zero.
 * model(Model) – pointer to the model structure
 * learning_rate(float) - learning rate
 * beta1(float) - momentum coefficient
 * beta2(float) - RMSprop coefficient
 * eps(float) - for numerical stability
 * clip_norm(float) - gradient clipping by norm
 * weight_decay(float) - reduces the model's weight values ​​at each training step
 */
float OptimizerStep(
	Model* self, 
	const float learning_rate, 
	const float beta1, 
	const float beta2, 
	const float eps, 
	const float clip_norm,
	const float weight_decay
);

/*
 * Resets all optimizer gradients.
 * model – pointer to the model structure
 */
void OptimizerReset(
	Model* model
);
#endif 
/*
 * Attaches to the model a tokenizer that was created previously, outside of this model.
 * model – pointer to the model structure
 * tokenizer – pointer to the tokenizer structure
 * IMPORTANT:
 *   - The model takes ownership. FreeModel() will call tok_destroy().
 *   - Do NOT free the tokenizer yourself after this call.
 *   - Do NOT attach the same tokenizer to two models.
 *   - If the model already holds a different tokenizer, it will be
 *     destroyed automatically before attaching the new one.
 */
void AttachTokenizer(
	Model* model,
	Tokenizer* tokenizer
);

void ShowDetails(Model* self);

/*
 * A function for saving the full model with all gradients, or for saving only the configuration and weights.
 * model - a pointer to the model to save
 * with_gradients  - condition parameter: if 0, it saves only the configuration and weights; otherwise, it saves the entire model (configuration, weights, and gradients).
 * filepath - path for saving the file on the system
 * return: int -> 1 error, 0 succes
 */
int SaveModel(const Model* model, const int with_gradients, const char* filepath);

/*
 * Creates a model and loads its weights from a file.
 * filepath - path to a model file
 * use_mmap - 0 = allocate + read (portable);
 *            1 = mmap file (POSIX only, ignored on Windows)
 * Returns: pointer to the created model on success, NULL on error.
 */
Model* LoadModel(const char* filepath, const int use_mmap);

/*
 * Complete removal of the model and resetting the pointer
 * self – a pointer to a variable that contains a pointer to a model
 */
void FreeModel(Model** self);