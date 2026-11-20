#include "engine.h"
#include "tokenizer.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <errno.h>
#include <omp.h>
#ifdef _WIN32
// ¯\_(ツ)_/¯
#include "mman.h"
#else
#include <sys/mman.h>   
#include <unistd.h>
#endif


#define RAND_MAX_CUSTOM 32768   // or 32767.0f

static inline int rand_r(unsigned int* seed) {
	*seed = *seed * 1103515245 + 12345;
	return (int)(*seed / 65536) % RAND_MAX_CUSTOM; // Return from 0 to 32767 (corresponds RAND_MAX)
}

static inline void randh(float* arr, const size_t size, const float stddev) {
#pragma omp parallel
	{
		unsigned int seed = (unsigned int)time(NULL)
			^ (omp_get_thread_num() << 16);

#pragma omp for
		for (size_t i = 0; i < size; i++) {
			// Unique seed for random
			// Box-Muller transformation to form a normal distribution
			float u1 = (float)rand_r(&seed) / RAND_MAX_CUSTOM;
			float u2 = (float)rand_r(&seed) / RAND_MAX_CUSTOM;
			if (u1 < 1e-7f) u1 = 1e-7f;
			float z0 = sqrtf(-2.0f * logf(u1))
				* cosf(2.0f * 3.1415926535f * u2);
			arr[i] = z0 * stddev;
		}
	}
}
static inline void aprintf(const float* arr, const size_t rows, const size_t cols, const char* name) {
	if (!arr) return;
	printf("%s = [", name);
	for (size_t p = 0; p < rows * cols; p++) {
		if (p % cols == 0) {
			printf("\n    ");
		}
		printf("%.4f ", arr[p]);
	}
	printf("\n]\n");
}

int _build_type(void) {
#if MOUEN_TRAINING
	return 1;
#else
	return 0;
#endif
}

Model* CreateModel(
	const size_t vocab_size,
	const size_t position_size,
	const size_t d_model,
	const size_t num_heads,
	const size_t num_layers,
	const size_t ffn_dim,
	const float eps
) {
	// Protection against zero values
	if (vocab_size < 1 || position_size < 1 || d_model < 1 ||
		num_heads < 1 || num_layers < 1 || ffn_dim < 1) {
		printf("Incorrect dimensions specified\n");
		return NULL;
	}
	// Protection against fractional division for MultiHeadAttention
	if (d_model % num_heads != 0) {
		printf("d_model must be divisible by num_heads\n");
		return NULL;
	}

	Config* config = calloc(1, sizeof(*config));
	if (!config) {
		fprintf(stderr, "Error creating object: Config\n");
		return NULL;
	}

	// layer size
	size_t layer_size = 4 * d_model * d_model + 2 * d_model * ffn_dim + 4 * d_model;

	// head dim
	size_t head_dim = d_model / num_heads;

	// Saving the model settings.
	config->vocab_size = vocab_size;
	config->position_size = position_size;
	config->d_model = d_model;
	config->num_heads = num_heads;
	config->num_layers = num_layers;
	config->ffn_dim = ffn_dim;
	config->eps = eps;
	config->head_dim = head_dim;
	config->layer_size = layer_size;
	config->head_size = d_model * head_dim;

	size_t size_model = vocab_size * d_model +
						position_size * d_model +
						layer_size * num_layers +
						d_model * vocab_size;

	Weights* weights = calloc(1, sizeof(*weights));
	if (!weights) {
		free(config);
		fprintf(stderr, "Error creating object: Weights\n");
		return NULL;
	}
	weights->size_model = size_model;

	Workspace* workspace = calloc(1, sizeof(*workspace));
	if (!workspace) {
		free(weights);
		free(config);
		fprintf(stderr, "Error creating object: Workspace\n");
		return NULL;
	}

	Model* model = calloc(1, sizeof(*model));
	if (!model) {
		free(workspace);
		free(weights);
		free(config);
		fprintf(stderr, "Error creating object: Model\n");
		return NULL;
	}

	model->tokenizer = NULL;
	model->config = config;
	model->weights = weights;
	model->workspace = workspace;
#if MOUEN_TRAINING
	model->adamw = NULL;
#endif

	return model;
}


void FreeModel(Model** model) {
	if (!model || !*model) return;
	Model* m = *model;
	if (m->config) free(m->config);
	if (m->weights) {
		if (m->weights->is_mmap) {
#ifndef _WIN32
			if (m->weights->mmap_ptr) {
				munmap(m->weights->mmap_ptr, m->weights->size_mmap);
				m->weights->data = NULL;
			}
#endif
		} else {
			if (m->weights->data) free(m->weights->data);
		}
#if MOUEN_TRAINING
		if (m->weights->grad) free(m->weights->grad);
#endif
		free(m->weights);
	}
	if (m->workspace) {
		if (m->workspace->forward) free(m->workspace->forward);
		if (m->workspace->updated) free(m->workspace->updated);
#if MOUEN_TRAINING
		if (m->workspace->backward) free(m->workspace->backward);
#endif
		free(m->workspace);
	}
#if MOUEN_TRAINING
	if (m->adamw) {
		if (m->adamw->momentum) free(m->adamw->momentum);
		if (m->adamw->rmsprop) free(m->adamw->rmsprop);
		free(m->adamw);
	}
#endif
	if (m->tokenizer) tok_destroy(m->tokenizer);
	free(m);
	*model = NULL;
}

void ShowDetails(Model* self) {
	if (!self) return;
	if (!self->config || !self->weights) return;
	Config* config = self->config;
	Weights* weights = self->weights;
	Workspace* workspace = self->workspace;
	printf("### Details:\n");
	printf(" # Config:\n");
	printf("  - vocab_size(user): %zu\n", config->vocab_size);
	printf("  - position_size(user): %zu\n", config->position_size);
	printf("  - d_model(user): %zu\n", config->d_model);
	printf("  - num_heads(user): %zu\n", config->num_heads);
	printf("  - head_dim(automatically): %zu\n", config->head_dim);
	printf("  - num_layers(user): %zu\n", config->num_layers);
	printf("  - ffn_dim(user): %zu\n", config->ffn_dim);
	printf("  - layer_size(automatically): %zu\n", config->layer_size);
	printf("  - head_size(automatically): %zu\n", config->head_size);
	printf(" # Weights:\n");
	printf("  - total: %zu\n", weights->size_model);
	printf("  - data pointer: %p\n", weights->data);
	aprintf(self->weights->data, 1, 5, "  * first 5");
#if MOUEN_TRAINING
	printf("  - grad pointer: %p\n", self->weights->grad);
	aprintf(self->weights->grad, 1, 5, "  * first 5");
#endif
	printf(" # Workspace:\n");
	printf("  - forward pointer: %p\n", self->workspace->forward);
	aprintf(self->workspace->forward, 1, 5, "  * first 5");
#if MOUEN_TRAINING
	printf("  - backward pointer: %p\n", self->workspace->backward);
	aprintf(self->workspace->backward, 1, 5, "  * first 5");
	{
		// import settings
		size_t d_mod = config->d_model;
		size_t vocd = config->vocab_size;
		size_t posd = config->position_size;
		size_t ffd = config->ffn_dim;
		size_t head_dim = config->head_dim;
		size_t num_layers = config->num_layers;
		size_t size_model = weights->size_model;

		size_t total = size_model * 4;
		printf(" # expected amount of allocated memory:\n");
		printf("  * for training:\n");
		printf("    - parameters: %zu elements, %zu bytes\n", size_model, size_model * sizeof(*weights->data));
		printf("    - gradients: %zu elements, %zu bytes\n", size_model, size_model * sizeof(*weights->data));
		printf("    - momentum: %zu elements, %zu bytes\n", size_model, size_model * sizeof(*weights->data));
		printf("    - rmsprop: %zu elements, %zu bytes\n", size_model, size_model * sizeof(*weights->data));
		{
			size_t layer_forward_size = posd * d_mod * 7 + // Q/K/V/O/concat/Output/Residual
				posd * ffd * 2 + // FF_pre_act/FF1
				posd * d_mod * 2 + // FF2//Residual
				posd * 4 + // inv_std/mean x 2
				posd * d_mod * 4; // x_hat/ln_out x 2

			size_t size_forward = posd * d_mod + // encoding
				posd * posd + // Q@K.T matrix
				num_layers * layer_forward_size +
				posd * vocd; // lm head
			printf("    - forward: %zu elements, %zu bytes\n", size_forward, size_forward * sizeof(*workspace->forward));
			total += size_forward;
		}
		{
			size_t MAX_THREADS_FOR_ALLOCATED = (size_t)omp_get_max_threads();

			size_t backward_size = posd * vocd + // dlogits
				posd * d_mod * 3 + // dY/layernorm/dconcat
				MAX_THREADS_FOR_ALLOCATED * d_mod * 2 + // thread_gamma/thread_beta
				posd * head_dim * 4 + // dO_head/dQ/dK/dV
				posd * posd + // datt
				posd * ffd; // dffn1

			printf("    - backward: %zu elements, %zu bytes\n", backward_size, backward_size * sizeof(*workspace->backward));

			total += backward_size;
		}
		printf("    - total: %zu elements\n", total);
		double bytes = (double)total * sizeof(*weights->data);
		printf("    - total: %zu elements\n", total);
		printf("      - %.2f bytes\n", bytes);
		printf("      - %.2f KB\n", bytes / 1024.0);
		printf("      - %.2f MB\n", bytes / (1024.0 * 1024.0));
		printf("      - %.2f GB\n", bytes / (1024.0 * 1024.0 * 1024.0));

	}
#endif
	fflush(stdout);
}

static inline void set_array(
	float* restrict arr,
	float** restrict link,
	size_t* bias,
	const size_t step,
	const float stddev
) {
	randh(arr, step, stddev);
	if (link) *link = arr;
	*bias += step;
}

void InitModel(
	Model* model,
	const float stddev
) {
	if (!model) return;
	Config* config = model->config;
	Weights* weights = model->weights;

	// import settings
	size_t d_mod = config->d_model;
	size_t vocd = config->vocab_size;
	size_t posd = config->position_size;
	size_t ffd = config->ffn_dim;
	size_t size_model = weights->size_model;

	weights->data = calloc(size_model, sizeof(*weights->data));
	if (!weights->data) {
		fprintf(stderr, "Failed to allocate memory for the data array.\n");
		return;
	}

	{
		// Residual scale:
		float res_scale = 1.0f / sqrtf(2.0f * config->num_layers);

		size_t bias = 0;
		// embedding
		set_array(&weights->data[bias], &weights->embedding, &bias, vocd * d_mod, stddev);
		// positional
		set_array(&weights->data[bias], &weights->positional, &bias, posd * d_mod, stddev * 0.5f);
		// layers
		for (size_t layer = 0; layer < config->num_layers; layer++) {
			// gamma
			{
				float* restrict gamma = &weights->data[bias];
				for (size_t i = 0; i < d_mod; i++) {
					gamma[i] = 1.0f;
				}
				if (layer == 0) weights->gamma1 = gamma;
				bias += d_mod;
			}
			// beta
			if (layer == 0) weights->beta1 = &weights->data[bias];
			bias += d_mod;
			// w_q
			set_array(&weights->data[bias], (layer == 0) ? &weights->W_q : NULL, &bias, d_mod * d_mod, stddev);
			// w_k
			set_array(&weights->data[bias], (layer == 0) ? &weights->W_k : NULL, &bias, d_mod * d_mod, stddev);
			// w_v
			set_array(&weights->data[bias], (layer == 0) ? &weights->W_v : NULL, &bias, d_mod * d_mod, stddev);
			// w_o
			set_array(&weights->data[bias], (layer == 0) ? &weights->W_o : NULL, &bias, d_mod * d_mod, stddev * res_scale);
			// gamma
			{
				float* restrict gamma = &weights->data[bias];
				for (size_t i = 0; i < d_mod; i++) {
					gamma[i] = 1.0f;
				}
				if (layer == 0) weights->gamma2 = gamma;
				bias += d_mod;
			}
			// beta
			if (layer == 0) weights->beta2 = &weights->data[bias];
			bias += d_mod;
			// ffn1
			set_array(&weights->data[bias], (layer == 0) ? &weights->W_ffn1 : NULL, &bias, d_mod * ffd, stddev);
			// ffn2
			set_array(&weights->data[bias], (layer == 0) ? &weights->W_ffn2 : NULL, &bias, d_mod * ffd, stddev * res_scale);
		}

		set_array(&weights->data[bias], &weights->lm_head, &bias, d_mod * vocd, stddev);
	}
}

int InitWorkspace(
	Model* model,
	WorkspaceType type,
	size_t chunk_size
) {
	if (!model) return 1;

	Config* config = model->config;
	Workspace* workspace = model->workspace;

	// import settings
	size_t d_mod = config->d_model;
	size_t vocd = config->vocab_size;
	size_t posd = config->position_size;
	size_t ffd = config->ffn_dim;
	size_t num_layers = config->num_layers;
	size_t head_dim = config->head_dim;

	if (workspace->mode != type) {
		if (workspace->forward) { free(workspace->forward); workspace->forward = NULL; }
#if MOUEN_TRAINING
		if (workspace->backward) { free(workspace->backward); workspace->backward = NULL; }
#endif // MOUEN_TRAINING
		if (workspace->updated) { free(workspace->updated); workspace->updated = NULL; }
	}

	{
		switch (type) {
			case ENGINE_INFERENCE: {
				if (chunk_size == 0 || chunk_size > posd) {
					fprintf(stderr, "Cannot use the x parameter with this value: %zu\n", chunk_size);
					return 1;
				}
				size_t layer_inf_size = posd * d_mod * 2; // K_cache/V_cache
				size_t size_inference = d_mod * chunk_size * 3 + // x, y, heads_out
										num_layers * layer_inf_size + // KV_cache
										posd * chunk_size + // attention: Softmax(Q @ K^T) matrix
										head_dim * chunk_size + // Q chunk
										ffd * chunk_size + // ffd chunk
										vocd; // output(ONLY vector)
#ifdef DEBUG
				printf("inference: layer: %zu, model: %zu\n", layer_inf_size, size_inference);
#endif
				if (!workspace->updated) {
					workspace->updated = calloc(posd, sizeof(*workspace->updated));
					if (!workspace->updated) {
						fprintf(stderr, "Failed to allocate memory for the context array.\n");
						return 1;
					}

#pragma omp parallel for
					for (size_t i = 0; i < posd; i++) {
						workspace->updated[i] = SIZE_MAX;
					}
				}

				if (!workspace->forward) {
					workspace->layer_inference_size = layer_inf_size;
					workspace->forward = calloc(size_inference, sizeof(*workspace->forward));
					if (!workspace->forward) {
						fprintf(stderr, "Failed to allocate memory for the forward array.\n");
						return 1;
					}
				}

				workspace->chunk_size = chunk_size;

				{
					size_t bias = 0;
					// encoding
					workspace->encoding = &workspace->forward[bias]; // reuses(x)
					bias += d_mod * chunk_size;
					// layernorm
					// multihead attention
					workspace->q = &workspace->forward[bias];
					bias += head_dim * chunk_size;
					workspace->att = &workspace->forward[bias];
					bias += posd * chunk_size;
					workspace->o = &workspace->forward[bias]; // reuses(y)
					bias += d_mod * chunk_size;
					workspace->attention = &workspace->forward[bias]; // reuses(heads_out)
					bias += d_mod * chunk_size;
					// feedforward
					workspace->ff_out1 = &workspace->forward[bias];
					bias += ffd * chunk_size;
					// lm head
					workspace->logits = &workspace->forward[bias];
					bias += vocd;
					// KV-cache
					workspace->k = &workspace->forward[bias]; // KV-cache
					bias += posd * d_mod;
					workspace->v = &workspace->forward[bias]; // KV-cache
					bias += posd * d_mod;
					// skip others layers
					bias += layer_inf_size * (num_layers - 1);

				}
				break;
			}
#if MOUEN_TRAINING
			case ENGINE_TRAINING: {
				size_t layer_inf_size = posd * d_mod * 7 + // Q/K/V/O/concat/Output/Residual
										posd * ffd * 2 + // FF_pre_act/FF1
										posd * d_mod * 2 + // FF2//Residual
										posd * 4 + // inv_std/mean x 2
										posd * d_mod * 4; // x_hat/ln_out x 2

				size_t size_inference = posd * d_mod + // encoding
										posd * posd + // Q@K.T matrix
										num_layers * layer_inf_size + 
										posd * vocd; // lm head
#ifdef DEBUG
				printf("training: layer: %zu, model: %zu\n", layer_inf_size, size_inference);
#endif

				if (!workspace->forward) {
					workspace->layer_inference_size = layer_inf_size;
					workspace->forward = calloc(size_inference, sizeof(*workspace->forward));
					if (!workspace->forward) {
						fprintf(stderr, "Failed to allocate memory for the forward array.\n");
						return 1;
					}
				}


				{
					size_t bias = 0;
					// encoding
					workspace->encoding = &workspace->forward[bias];
					bias += posd * d_mod;
					// attention matrix
					workspace->att = &workspace->forward[bias];
					bias += posd * posd;
					// layernorm
					workspace->mean1 = &workspace->forward[bias];
					bias += posd;
					workspace->inv_std1 = &workspace->forward[bias];
					bias += posd;
					workspace->x_hat1 = &workspace->forward[bias];
					bias += posd * d_mod;
					workspace->layernorm1 = &workspace->forward[bias];
					bias += posd * d_mod;
					// multihead attention
					workspace->q = &workspace->forward[bias];
					bias += posd * d_mod;
					workspace->k = &workspace->forward[bias];
					bias += posd * d_mod;
					workspace->v = &workspace->forward[bias];
					bias += posd * d_mod;
					workspace->o = &workspace->forward[bias];
					bias += posd * d_mod;
					workspace->concat = &workspace->forward[bias];
					bias += posd * d_mod;
					workspace->attention = &workspace->forward[bias];
					bias += posd * d_mod;
					workspace->att_residual = &workspace->forward[bias];
					bias += posd * d_mod;
					// layernorm
					workspace->mean2 = &workspace->forward[bias];
					bias += posd;
					workspace->inv_std2 = &workspace->forward[bias];
					bias += posd;
					workspace->x_hat2 = &workspace->forward[bias];
					bias += posd * d_mod;
					workspace->layernorm2 = &workspace->forward[bias];
					bias += posd * d_mod;
					// feedforward
					workspace->ff_pre_act = &workspace->forward[bias];
					bias += posd * ffd;
					workspace->ff_out1 = &workspace->forward[bias];
					bias += posd * ffd;
					workspace->ff_out2 = &workspace->forward[bias];
					bias += posd * d_mod;
					workspace->ff_residual = &workspace->forward[bias];
					bias += posd * d_mod;
					// skip others layers
					bias += layer_inf_size * (num_layers - 1);
					// lm head
					workspace->logits = &workspace->forward[bias];
					bias += posd * vocd;
				}
				{
					size_t MAX_THREADS_FOR_ALLOCATED = (size_t)omp_get_max_threads();
					workspace->num_threads = MAX_THREADS_FOR_ALLOCATED;

					size_t backward_size = posd * vocd + // dlogits
						posd * d_mod * 3 + // dY/layernorm/dconcat
						MAX_THREADS_FOR_ALLOCATED * d_mod * 2 + // thread_gamma/thread_beta
						posd * head_dim * 4 + // dO_head/dQ/dK/dV
						posd * posd + // datt
						posd * ffd; // dffn1


					if (!workspace->backward) {
						workspace->backward = calloc(backward_size, sizeof(*workspace->backward));
						if (!workspace->backward) {
							fprintf(stderr, "Failed to allocate memory for the backward array.\n");
							return 1;
						}
					}

					size_t bias = 0;
					workspace->dlogits = &workspace->backward[bias];
					bias += posd * vocd;
					workspace->dY = &workspace->backward[bias];
					bias += posd * d_mod;
					workspace->dffn1 = &workspace->backward[bias];
					bias += posd * ffd;
					workspace->dlayernorm = &workspace->backward[bias];
					bias += posd * d_mod;
					workspace->dconcat = &workspace->backward[bias];
					bias += posd * d_mod;
					workspace->dO_head = &workspace->backward[bias];
					bias += posd * head_dim;
					workspace->datt = &workspace->backward[bias];
					bias += posd * posd;
					workspace->dV = &workspace->backward[bias];
					bias += posd * head_dim;
					workspace->dK = &workspace->backward[bias];
					bias += posd * head_dim;
					workspace->dQ = &workspace->backward[bias];
					bias += posd * head_dim;
					workspace->thread_gamma = &workspace->backward[bias];
					bias += MAX_THREADS_FOR_ALLOCATED * d_mod;
					workspace->thread_beta = &workspace->backward[bias];
					bias += MAX_THREADS_FOR_ALLOCATED * d_mod;
				}
				break;
			}
#endif
			default:
				fprintf(stderr, "Unknown operating mode\n");
				return 1;
		}
	}
	workspace->mode = type;

	return 0;
}

static inline void matmul(
	const float* restrict a,
	const float* restrict b,
	float* restrict out,
	const size_t a_rows,
	const size_t b_rows,
	const size_t b_cols
) {
#pragma omp for
	for (size_t i = 0; i < a_rows; i++) {
		const float* restrict a_row = &a[i * b_rows];
		float* restrict out_row = &out[i * b_cols];

		for (size_t j = 0; j < b_rows; j++) {
			const float a_val = a_row[j];
			const float* restrict b_row = &b[j * b_cols];

#pragma omp simd
			for (size_t k = 0; k < b_cols; k++) {
				out_row[k] += a_val * b_row[k];
			}
		}
	}
}

static inline void layernorm_inference(
	const size_t inputs_size,
	const size_t d_mod,
	const float d,
	const float eps,
	const float* restrict layer_input,
	const float* restrict gamma,
	const float* restrict beta,
	float* restrict l_ln_out
) {
#pragma omp for
	for (size_t i = 0; i < inputs_size; i++) {
		size_t pos = i * d_mod;
		const float* restrict inp_row = &layer_input[pos];
		float* restrict out_row = &l_ln_out[pos];
		// mean
		float mean = 0.0f;
#pragma omp simd reduction(+:mean)
		for (size_t j = 0; j < d_mod; j++) {
			mean += inp_row[j];
		}
		mean *= d;

		// inv_std
		float var = 0.0f;
#pragma omp simd reduction(+:var)
		for (size_t j = 0; j < d_mod; j++) {
			const float x = inp_row[j] - mean;
			var += x * x;
		}
		var *= d;
		var = 1.0f / sqrtf(var + eps);

#pragma omp simd
		for (size_t j = 0; j < d_mod; j++) {
			float x_now = (inp_row[j] - mean) * var;
			out_row[j] = gamma[j] * x_now + beta[j];
		}
	}
}

// TODO: Remove the -inf mask, compute the Softmax along the 'end' dimension, and then zero out the remaining values ​​after all operations are complete.
static inline void Scaled_Dot_Product_Attention(
	const float* restrict a,
	const float* restrict b_T,
	float* restrict out,
	const size_t a_rows,
	const size_t a_cols,
	const size_t b_rows,
	size_t pos
) {
	const float scale = 1.0f / sqrtf((float)a_cols);

#pragma omp for schedule(guided)
	for (size_t i = 0; i < a_rows; i++) {

		float max = -INFINITY;
		const size_t end = (pos + i + 1 < b_rows) ? (pos + i + 1) : b_rows;

		float* restrict out_row = &out[i * b_rows];
		const float* restrict a_row = &a[i * a_cols];
		// Q@K.T
		for (size_t k = 0; k < end; k++) {
			const float* restrict b_row = &b_T[k * a_cols];

			float sum = 0.0f;
#pragma omp simd reduction(+:sum)
			for (size_t j = 0; j < a_cols; j++) {
				sum += a_row[j] * b_row[j];
			}
			// Scaling
			const float now = sum * scale;
			out_row[k] = now;
			if (now > max) max = now;
		}
		// Masking
#pragma omp simd
		for (size_t k = end; k < b_rows; k++) {
			out_row[k] = -INFINITY;
		}
		// Softmax sum
		float sum = 0.0f;
#pragma omp simd reduction(+:sum)
		for (size_t k = 0; k < b_rows; k++) {
			const float x = expf(out_row[k] - max);
			out_row[k] = x;
			sum += x;
		}

		// Softmax scaling
		sum = (1.0f / sum);
#pragma omp simd
		for (size_t k = 0; k < b_rows; k++) {
			out_row[k] *= sum;
		}
	}
}

size_t InferenceStep(
	Model* model,
	const size_t* inputs,
	const size_t inputs_size,
	const float temperature
) {
	if (!model || !inputs || inputs_size == 0) return SIZE_MAX;

	Config* config = model->config;
	Weights* weights = model->weights;
	Workspace* workspace = model->workspace;

	if (workspace->mode != ENGINE_INFERENCE) {
		fprintf(stderr, "InferenceStep requires ENGINE_INFERENCE workspace\n");
		return SIZE_MAX;
	}
	if (!workspace->updated) {
		fprintf(stderr, "Workspace not initialized (updated == NULL)\n");
		return SIZE_MAX;
	}

	// Config settings
	const size_t d_mod = config->d_model;
	const size_t head_dim = config->head_dim;
	const size_t vocd = config->vocab_size;
	const size_t posd = config->position_size;
	const size_t ffd = config->ffn_dim;
	const size_t layer_size = config->layer_size;
	const size_t num_layers = config->num_layers;
	const size_t num_heads = config->num_heads;
	const size_t head_size = config->head_size;
	const float eps = config->eps;

	// Weights data
	const float* restrict embedding = weights->embedding;
	const float* restrict positional = weights->positional;
	const float* restrict gamma1 = weights->gamma1;
	const float* restrict beta1 = weights->beta1;
	const float* restrict W_q = weights->W_q;
	const float* restrict W_k = weights->W_k;
	const float* restrict W_v = weights->W_v;
	const float* restrict W_o = weights->W_o;
	const float* restrict gamma2 = weights->gamma2;
	const float* restrict beta2 = weights->beta2;
	const float* restrict w_ffn1 = weights->W_ffn1;
	const float* restrict w_ffn2 = weights->W_ffn2;
	const float* restrict w_lmhead = weights->lm_head;

	// Workspace forward
	float* restrict x = workspace->encoding;
	float* restrict q = workspace->q;
	float* restrict k = workspace->k;
	float* restrict v = workspace->v;
	float* restrict att = workspace->att;
	float* restrict y = workspace->o;
	float* restrict heads_out = workspace->attention;
	float* restrict ff_out1 = workspace->ff_out1;
	float* restrict logits = workspace->logits;
	const size_t layer_inference_size = workspace->layer_inference_size;
	const size_t chunk_size = workspace->chunk_size;

	if (inputs_size > posd) {
		fprintf(stderr, "The context is overflowing\n");
		return SIZE_MAX;
	}

	// we are looking for a token that has no cache
	size_t* restrict updated = workspace->updated;
	size_t start_updated = inputs_size;
	for (size_t i = 0; i < inputs_size; i++) {
		if (updated[i] != inputs[i]) {
			start_updated = i;
			break;
		}
	}

	// If there are no new tokens, we return SIZE_MAX, since the prediction has not changed.
	if (start_updated == inputs_size) {
		return SIZE_MAX;
	}

	// we check that the tokens are in the vocabulary
	for (size_t i = start_updated; i < inputs_size; i++) {
		if (inputs[i] >= vocd) {
			fprintf(stderr, "Attempt to pass a non-existent token\n");
			return SIZE_MAX;
		}
	}

	// we update the processed tokens so that we don't count them in the future
	for (size_t i = start_updated; i < inputs_size; i++) {
		updated[i] = inputs[i];
	}

	size_t pos = start_updated;   // cursor
	while (pos < inputs_size) {
		size_t chunk = (pos + chunk_size < inputs_size) ? chunk_size : inputs_size - pos;
		int is_last = (pos + chunk == inputs_size);

#pragma omp parallel for
		for (size_t i = 0; i < chunk; i++) {
			size_t idx = i + pos;
			float* restrict x_row = &x[i * d_mod];
			const float* restrict emb_row = &embedding[inputs[idx] * d_mod];
			const float* restrict pos_row = &positional[idx * d_mod];
#pragma omp simd
			for (size_t j = 0; j < d_mod; j++) {
				x_row[j] = emb_row[j] + pos_row[j];
			}
		}

		for (size_t layer = 0; layer < num_layers; layer++) {
			const float d = 1.0f / (float)d_mod;
			size_t jump_inference = layer * layer_inference_size;
			size_t jump_param = layer * layer_size;
#pragma omp parallel
			{
				const float* restrict gamma = &gamma1[jump_param];
				const float* restrict beta = &beta1[jump_param];
				layernorm_inference(chunk, d_mod, d, eps, x, gamma, beta, y);
			}

			// multihead attention
			{
				// weights
				const float* restrict l_w_q = &W_q[jump_param];
				const float* restrict l_w_k = &W_k[jump_param];
				const float* restrict l_w_v = &W_v[jump_param];
				const float* restrict l_w_o = &W_o[jump_param];
				// buffers
				float* restrict l_k_cache = &k[jump_inference];
				float* restrict l_v_cache = &v[jump_inference];
				// [head00][head10][headx0]...[head0y][head1y][headxy]

				for (size_t head = 0; head < num_heads; head++) {
					size_t idx_w = head * head_size; //[head0][head1]...[headn]
					const float* restrict h_w_q = &l_w_q[idx_w];
					const float* restrict h_w_k = &l_w_k[idx_w];
					const float* restrict h_w_v = &l_w_v[idx_w];
					// (we get a pointer to the head)
					size_t idx_cache = head * posd * head_dim; //[head0][head1]...[headn]
					float* restrict h_k_cache = &l_k_cache[idx_cache];
					float* restrict h_v_cache = &l_v_cache[idx_cache];
					// (we apply an offset to update the required chunk)
					size_t idx_chunk = pos * head_dim;
					float* restrict h_k = &h_k_cache[idx_chunk];
					float* restrict h_v = &h_v_cache[idx_chunk];

					// QKV reset
					memset(q, 0, head_dim * chunk * sizeof(*q));
					memset(h_k, 0, head_dim * chunk * sizeof(*h_k));
					memset(h_v, 0, head_dim * chunk * sizeof(*h_v));
#pragma omp parallel
					{
						// QKV
						matmul(y, h_w_q, q, chunk, d_mod, head_dim); // (chunk, head_dim)
						matmul(y, h_w_k, h_k, chunk, d_mod, head_dim); // (chunk, head_dim)
						matmul(y, h_w_v, h_v, chunk, d_mod, head_dim); // (chunk, head_dim)

						// (chunk, head_dim) @ (pos + chunk, head_dim).T => (chunk, pos+chunk)
						Scaled_Dot_Product_Attention(q, h_k_cache, att, chunk, head_dim, pos + chunk, pos); 

						// reuse buffer q to multiply by v
#pragma omp for simd
						for (size_t i = 0; i < head_dim * chunk; i++) {
							q[i] = 0;
						}

						// att[chunk, pos+chunk] @ V[chunk+pos, head_dim] => [chunk, head_dim]
						matmul(att, h_v_cache, q, chunk, chunk + pos, head_dim); // (chunk, head_dim)

						// we immediately transpose the result for x @ W_o
						{
#pragma omp for 
							for (size_t i = 0; i < chunk; i++) {
								const float* restrict inp_row = &q[i * head_dim];
								float* restrict out_row = &heads_out[head * head_dim + i * d_mod];
#pragma omp simd
								for (size_t j = 0; j < head_dim; j++) {
									out_row[j] = inp_row[j];
								}
							}
						}
					}
				}
				memset(ff_out1, 0, chunk* ffd * sizeof(*ff_out1));
#pragma omp parallel
				{
					// concat(chunk, d_mod) @ W_o(d_mod, d_mod) += attention(chunk, d_mod)
					matmul(heads_out, l_w_o, x, chunk, d_mod, d_mod);
					// LayerNorm2
					{
						const float* restrict gamma = &gamma2[jump_param];
						const float* restrict beta = &beta2[jump_param];
						layernorm_inference(chunk, d_mod, d, eps, x, gamma, beta, y);
					}
					// FeedForward
					{
						const float* restrict l_w_ffn1 = &w_ffn1[jump_param]; // (d_mod, ffd)
						const float* restrict l_w_ffn2 = &w_ffn2[jump_param]; // (ffd, d_mod)

						// (chunk, d_mod) @ (d_mod, ffd) => (chunk, ffd)
						matmul(y, l_w_ffn1, ff_out1, chunk, d_mod, ffd);
#pragma omp for simd
						for (size_t idx = 0; idx < chunk * ffd; idx++) {
							const float x = ff_out1[idx];
							ff_out1[idx] = x / (1.0f + expf(-x));
						}
						// (chunk, ffd) @ (ffd, d_mod) => (chunk, d_mod)
						matmul(ff_out1, l_w_ffn2, x, chunk, ffd, d_mod);
					}
				}
			}

		}

		// lm_head
		float* restrict predict = &x[(chunk - 1) * d_mod];
		memset(logits, 0, vocd * sizeof(*logits));
		// (1, d_mod) @ (d_mod, vocd) => (1, vocd)
		for (size_t i = 0; i < d_mod; i++) {
			const float x_val = predict[i];
			const float* restrict lm_head_row = &w_lmhead[i * vocd];
#pragma omp simd
			for (size_t j = 0; j < vocd; j++) {
				logits[j] += x_val * lm_head_row[j];
			}
		}
		// softmax
		{
			float glob_max = -INFINITY;
			float sum = 0.0f;
			float scale = 1.0f / temperature;

#pragma omp parallel
			{
#pragma omp for reduction(max:glob_max)
				for (size_t i = 0; i < vocd; i++) {
					const float x = logits[i] * scale;
					logits[i] = x;
					if (x > glob_max) glob_max = x;
				}


#pragma omp for reduction(+:sum)
				for (size_t i = 0; i < vocd; i++) {
					const float x = expf(logits[i] - glob_max);
					logits[i] = x;
					sum += x;
				}

				float inv_sum = 1.0f / sum;

#pragma omp for simd
				for (size_t i = 0; i < vocd; i++) {
					logits[i] *= inv_sum;
				}
			}
		}

		pos += chunk;
	}

	// Temperature sampling: sample from the distribution
	unsigned int seed = (unsigned int)time(NULL) ^ (omp_get_thread_num() << 16) ^ (unsigned int)clock() ^ (unsigned int)(inputs_size);
	float r = (float)rand_r(&seed) / RAND_MAX_CUSTOM;
	float cum = 0.0f;
	for (size_t i = 0; i < vocd; i++) {
		cum += logits[i];
		if (r < cum) {
			return i;
		}
	}
	// Fallback (should never happen)
	return SIZE_MAX;
}

#if MOUEN_TRAINING
int InitTrain(
	Model* model
) {
	if (!model) return 1;
	Config* config = model->config;
	Weights* weights = model->weights;

	if (!model->adamw) {
		model->adamw = calloc(1, sizeof(*model->adamw));
		if (!model->adamw) {
			fprintf(stderr, "Failed to allocate memory for the adamw object.\n");
			return 1;
		}
	}
	AdamW* adamw = model->adamw;

	// allocated for adamw
	{
		size_t size_model = weights->size_model;

		if (!adamw->momentum) {
			adamw->momentum = calloc(size_model, sizeof(*adamw->momentum));
			if (!adamw->momentum) {
				fprintf(stderr, "Failed to allocate memory for the momentum array.\n");
				return 1;
			}
		}

		if (!adamw->rmsprop) {
			adamw->rmsprop = calloc(size_model, sizeof(*adamw->rmsprop));
			if (!adamw->rmsprop) {
				fprintf(stderr, "Failed to allocate memory for the rmsprop array.\n");
				return 1;
			}
		}
	}

	// import settings
	size_t d_mod = config->d_model;
	size_t vocd = config->vocab_size;
	size_t posd = config->position_size;
	size_t ffd = config->ffn_dim;
	size_t layer_size = config->layer_size;

	// Gradients
	if (!weights->grad) {
		weights->grad = calloc(weights->size_model, sizeof(*weights->grad));
		if (!weights->grad) {
			fprintf(stderr, "Failed to allocate memory for the grad array.\n");
			return 1;
		}
	}

	{
		size_t bias = 0;

		// embedding
		weights->grad_embedding = &weights->grad[bias];
		bias += vocd * d_mod;
		// positional
		weights->grad_positional = &weights->grad[bias];
		bias += posd * d_mod;
		// gamma1
		weights->grad_gamma1 = &weights->grad[bias];
		bias += d_mod;
		// beta1
		weights->grad_beta1 = &weights->grad[bias];
		bias += d_mod;
		// W_q
		weights->grad_W_q = &weights->grad[bias];
		bias += d_mod * d_mod;
		// W_k
		weights->grad_W_k = &weights->grad[bias];
		bias += d_mod * d_mod;
		// W_v
		weights->grad_W_v = &weights->grad[bias];
		bias += d_mod * d_mod;
		// W_o
		weights->grad_W_o = &weights->grad[bias];
		bias += d_mod * d_mod;
		// gamma2
		weights->grad_gamma2 = &weights->grad[bias];
		bias += d_mod;
		// beta2
		weights->grad_beta2 = &weights->grad[bias];
		bias += d_mod;
		// ffn1
		weights->grad_W_ffn1 = &weights->grad[bias];
		bias += d_mod * ffd;
		// ffn2
		weights->grad_W_ffn2 = &weights->grad[bias];
		bias += d_mod * ffd;
		// others
		bias += layer_size * (config->num_layers - 1);
		// lm_head
		weights->grad_lm_head = &weights->grad[bias];
	}
	return 0;
}

void OptimizerReset(
	Model* model
) {
	if (!model || !model->adamw || !model->weights || !model->workspace) return;
	Weights* weights = model->weights;
	AdamW* adamw = model->adamw;
	Workspace* workspace = model->workspace;

	size_t size = weights->size_model;

	adamw->step = 0;

	if (weights->grad) memset(weights->grad, 0, size * sizeof(*weights->grad));
	if (adamw->momentum) memset(adamw->momentum, 0, size * sizeof(*adamw->momentum));
	if (adamw->rmsprop) memset(adamw->rmsprop, 0, size * sizeof(*adamw->rmsprop));

	workspace->batchs = 0;
}

static inline void layernorm_inf_train(
	const size_t inputs_size,
	const size_t d_mod,
	const float d,
	const float eps,
	const float* restrict layer_input,
	const float* restrict gamma,
	const float* restrict beta,
	float* restrict l_mean,
	float* restrict l_inv_std,
	float* restrict l_x_hat,
	float* restrict l_ln_out
) {
#pragma omp for
	for (size_t i = 0; i < inputs_size; i++) {
		size_t pos = i * d_mod;
		const float* restrict inp_row = &layer_input[pos];
		float* restrict x_hat_row = &l_x_hat[pos];
		float* restrict out_row = &l_ln_out[pos];
		// mean
		float mean = 0.0f;
#pragma omp simd reduction(+:mean)
		for (size_t j = 0; j < d_mod; j++) {
			mean += inp_row[j];
		}
		mean *= d;
		l_mean[i] = mean;

		// inv_std
		float var = 0.0f;
#pragma omp simd reduction(+:var)
		for (size_t j = 0; j < d_mod; j++) {
			const float x = inp_row[j] - mean;
			var += x * x;
		}
		var *= d;
		var = 1.0f / sqrtf(var + eps);
		l_inv_std[i] = var;

#pragma omp simd
		for (size_t j = 0; j < d_mod; j++) {
			float x_now = (inp_row[j] - mean) * var;
			x_hat_row[j] = x_now;
			out_row[j] = gamma[j] * x_now + beta[j];
		}
	}
}

// TODO: Fix cache-miss in the future.
static inline void matmulT_(
	const float* restrict a_T,
	const float* restrict b,
	float* restrict out,
	const size_t a_rows,
	const size_t a_cols,
	const size_t b_cols
) {
#pragma omp for
	for (size_t j = 0; j < a_cols; j++) {
		float* restrict _out_row = &out[j * b_cols];
		for (size_t i = 0; i < a_rows; i++) {
			float _a_val = a_T[i * a_cols + j];
			const float* restrict _b_row = &b[i * b_cols];
#pragma omp simd
			for (size_t k = 0; k < b_cols; k++) {
				_out_row[k] += _a_val * _b_row[k];
			}
		}
	}
}

static inline void matmul_T(
	const float* restrict a,
	const float* restrict b_T,
	float* restrict out,
	const size_t a_rows,
	const size_t a_cols,
	const size_t b_rows
) {

#pragma omp for
	for (size_t i = 0; i < a_rows; i++) {
		float* restrict _out_row = &out[i * b_rows];
		const float* restrict a_row = &a[i * a_cols];
		for (size_t k = 0; k < b_rows; k++) {
			const float* restrict b_row = &b_T[k * a_cols];

			float sum = 0.0f;
#pragma omp simd reduction(+:sum)
			for (size_t j = 0; j < a_cols; j++) {
				sum += a_row[j] * b_row[j];
			}
			_out_row[k] += sum;
		}
	}
}

// TODO: Add protection against changes to the number of threads.
// Does not set the gradient, but accumulates it.
static inline void layernorm_grad(
	const float* restrict gradient_array,
	const float* restrict x_hat,
	const float* restrict inv_std,
	const float* restrict gamma_array,
	float* restrict return_gradient,
	float* restrict threads_gamma,
	float* restrict threads_beta,
	float* restrict dgamma,
	float* restrict dbeta,
	const size_t inps,
	const size_t d_model
) {
	// LayerNorm backprop
	// cache-friendly && parallel loop.

	const float inv_d = 1.0f / (float)d_model;

	// Local cache
	int tid = omp_get_thread_num(); // actual_thread
	float* restrict local_gamma = &threads_gamma[tid * d_model];
	float* restrict local_beta = &threads_beta[tid * d_model];
	memset(local_gamma, 0, d_model * sizeof(*local_gamma));
	memset(local_beta, 0, d_model * sizeof(*local_beta));

#pragma omp for
	for (size_t i = 0; i < inps; i++) {


		// For gamma and beta
		const float* restrict dln_row = &gradient_array[i * d_model];
		const float* restrict x_hat_row = &x_hat[i * d_model];

		// For dLayerNorm
		float* restrict dx_row = &return_gradient[i * d_model];
		const float inv_std_local = inv_std[i];
		float sum_dx_hat = 0.0f;
		float sum_dx_hat_x = 0.0f;
		// if (isinf(inv_std_local)) printf("std: pointer: %p\n", inv_std);

#pragma omp simd reduction(+:sum_dx_hat, sum_dx_hat_x)
		for (size_t j = 0; j < d_model; j++) {
			// for sum
			float dx_hat = dln_row[j] * gamma_array[j];

			sum_dx_hat += dx_hat;
			sum_dx_hat_x += dx_hat * x_hat_row[j];
		}
#pragma omp simd
		for (size_t j = 0; j < d_model; j++) {
			const float dln_val = dln_row[j];
			float dx_hat = dln_val * gamma_array[j];

			float dx = (dx_hat - (sum_dx_hat + x_hat_row[j] * sum_dx_hat_x) * inv_d) * inv_std_local;
			dx_row[j] += dx;

			local_gamma[j] += dln_val * x_hat_row[j];
			local_beta[j] += dln_val;
		}
	}

	for (size_t i = 0; i < d_model; i++) {
#pragma omp atomic
		dgamma[i] += local_gamma[i];
#pragma omp atomic
		dbeta[i] += local_beta[i];
	}
}

float TrainStep(
	Model* model,
	const size_t* inputs,
	const size_t* target,
	const uint8_t* labels,
	const size_t inputs_size
) {
	if (!model || !inputs || !target || !labels) {
		fprintf(stderr, "One of the parameters was not passed\n");
		return NAN;
	}
	if (inputs_size == 0 || inputs_size > model->config->position_size) return NAN;

	// Obtaining class pointers
	Config* config = model->config;
	Workspace* workspace = model->workspace;
	Weights* weights = model->weights;

	if (workspace->mode != ENGINE_TRAINING) {
		fprintf(stderr, "TrainStep requires ENGINE_TRAINING workspace\n");
		return NAN;
	}

	// Config settings
	const size_t d_mod = config->d_model;
	const size_t head_dim = config->head_dim;
	const size_t vocd = config->vocab_size;
	const size_t posd = config->position_size;
	const size_t ffd = config->ffn_dim;
	const size_t layer_size = config->layer_size;
	const size_t num_layers = config->num_layers;
	const size_t num_heads = config->num_heads;
	const size_t head_size = config->head_size;
	const float eps = config->eps;

	// Weights data
	const float* restrict embedding = weights->embedding;
	const float* restrict positional = weights->positional;
	const float* restrict gamma1 = weights->gamma1;
	const float* restrict beta1 = weights->beta1;
	const float* restrict W_q = weights->W_q;
	const float* restrict W_k = weights->W_k;
	const float* restrict W_v = weights->W_v;
	const float* restrict W_o = weights->W_o;
	const float* restrict gamma2 = weights->gamma2;
	const float* restrict beta2 = weights->beta2;
	const float* restrict w_ffn1 = weights->W_ffn1;
	const float* restrict w_ffn2 = weights->W_ffn2;
	const float* restrict w_lmhead = weights->lm_head;

	// Weights grad
	float* restrict grad_embedding = weights->grad_embedding;
	float* restrict grad_positional = weights->grad_positional;
	float* restrict grad_W_q = weights->grad_W_q;
	float* restrict grad_W_k = weights->grad_W_k;
	float* restrict grad_W_v = weights->grad_W_v;
	float* restrict grad_W_o = weights->grad_W_o;
	float* restrict grad_W_ffn1 = weights->grad_W_ffn1;
	float* restrict grad_W_ffn2 = weights->grad_W_ffn2;
	float* restrict grad_gamma1 = weights->grad_gamma1;
	float* restrict grad_beta1 = weights->grad_beta1;
	float* restrict grad_gamma2 = weights->grad_gamma2;
	float* restrict grad_beta2 = weights->grad_beta2;
	float* restrict grad_lm_head = weights->grad_lm_head;

	// Workspace forward
	float* restrict encoding = workspace->encoding;
	float* restrict mean1 = workspace->mean1;
	float* restrict inv_std1 = workspace->inv_std1;
	float* restrict x_hat1 = workspace->x_hat1;
	float* restrict ln1 = workspace->layernorm1;
	float* restrict q = workspace->q;
	float* restrict k = workspace->k;
	float* restrict v = workspace->v;
	float* restrict att = workspace->att;
	float* restrict o = workspace->o;
	float* restrict concat = workspace->concat;
	float* restrict attention = workspace->attention;
	float* restrict att_residual = workspace->att_residual;
	float* restrict mean2 = workspace->mean2;
	float* restrict inv_std2 = workspace->inv_std2;
	float* restrict x_hat2 = workspace->x_hat2;
	float* restrict ln2 = workspace->layernorm2;
	float* restrict ff_pre_act = workspace->ff_pre_act;
	float* restrict ff_out1 = workspace->ff_out1;
	float* restrict ff_out2 = workspace->ff_out2;
	float* restrict ff_residual = workspace->ff_residual;
	float* restrict logits = workspace->logits;
	const size_t layer_inference_size = workspace->layer_inference_size;

	// Workspace backward
	float* restrict dlogits = workspace->dlogits;
	float* restrict dY = workspace->dY;
	float* restrict dffn1 = workspace->dffn1;
	float* restrict dlayernorm = workspace->dlayernorm;
	float* restrict dconcat = workspace->dconcat;
	float* restrict dO_head = workspace->dO_head;
	float* restrict datt = workspace->datt;
	float* restrict dV = workspace->dV;
	float* restrict dK = workspace->dK;
	float* restrict dQ = workspace->dQ;
	float* restrict thread_gamma = workspace->thread_gamma;
	float* restrict thread_beta = workspace->thread_beta;

	// Checking tokens for out-of-bounds
	for (size_t i = 0; i < inputs_size; i++) {
		const size_t token = inputs[i];
		if (token >= vocd) return NAN;
	}
	// encoding (inps, d_mod)
#pragma omp  parallel for
	for (size_t i = 0; i < inputs_size; i++) {
		const float* restrict emb_row = &embedding[inputs[i] * d_mod];
		const float* restrict pos_row = &positional[i * d_mod];
		float* restrict enc_row = &encoding[i * d_mod];
#pragma omp simd
		for (size_t j = 0; j < d_mod; j++) {
			enc_row[j] = emb_row[j] + pos_row[j];
		}
	}
	float* restrict layer_input = encoding;
	// Layer Forward
	for (size_t layer = 0; layer < num_layers; layer++) {
		const float d = 1.0f / (float)d_mod;
		size_t jump_inference = layer * layer_inference_size;
		size_t jump_param = layer * layer_size;
		float* restrict l_ff_res = &ff_residual[jump_inference];
		// LayerNorm1
#pragma omp parallel
		{
			const float* restrict gamma = &gamma1[jump_param];
			const float* restrict beta = &beta1[jump_param];
			float* restrict l_mean = &mean1[jump_inference];
			float* restrict l_inv_std = &inv_std1[jump_inference];
			float* restrict l_x_hat = &x_hat1[jump_inference];
			float* restrict l_ln = &ln1[jump_inference];
			layernorm_inf_train(
				inputs_size, d_mod, d,
				eps, layer_input, gamma,
				beta, l_mean, l_inv_std,
				l_x_hat, l_ln
			);
		}
		// multihead attention
		{
			// inputs
			const float* restrict l_inp = &ln1[jump_inference];
			// weights
			const float* restrict l_w_q = &W_q[jump_param];
			const float* restrict l_w_k = &W_k[jump_param];
			const float* restrict l_w_v = &W_v[jump_param];
			const float* restrict l_w_o = &W_o[jump_param];
			// buffers
			float* restrict l_q = &q[jump_inference];
			float* restrict l_k = &k[jump_inference];
			float* restrict l_v = &v[jump_inference];
			float* restrict l_o = &o[jump_inference];
			float* restrict l_concat = &concat[jump_inference];
			float* restrict l_out = &attention[jump_inference];
			float* restrict l_res = &att_residual[jump_inference];


			size_t reset_size = posd * d_mod;
			memset(l_q, 0, reset_size * sizeof(*l_q));
			memset(l_k, 0, reset_size * sizeof(*l_k));
			memset(l_v, 0, reset_size * sizeof(*l_v));
			memset(l_o, 0, reset_size * sizeof(*l_o));
			memset(l_out, 0, inputs_size * d_mod * sizeof(*l_out));
			for (size_t head = 0; head < num_heads; head++) {
#pragma omp parallel
				{
					size_t idx_w = head * head_size; //[head0][head1]...[headn]
					size_t idx_f = head * posd * head_dim; //[head0][head1]...[headn]
					const float* restrict h_w_q = &l_w_q[idx_w];
					const float* restrict h_w_k = &l_w_k[idx_w];
					const float* restrict h_w_v = &l_w_v[idx_w];
					float* restrict h_q = &l_q[idx_f];
					float* restrict h_k = &l_k[idx_f];
					float* restrict h_v = &l_v[idx_f];
					float* restrict h_o = &l_o[idx_f];

					// QKV
					matmul(l_inp, h_w_q, h_q, inputs_size, d_mod, head_dim); // (inps, head_dim)
					matmul(l_inp, h_w_k, h_k, inputs_size, d_mod, head_dim); // (inps, head_dim)
					matmul(l_inp, h_w_v, h_v, inputs_size, d_mod, head_dim); // (inps, head_dim)


					Scaled_Dot_Product_Attention(h_q, h_k, att, inputs_size, head_dim, inputs_size, 0); // (inps, inps)

					// x[inps, inps] @ V[inps, head_dim] => [inps, head_dim]
					matmul(att, h_v, h_o, inputs_size, inputs_size, head_dim); // (inps, head_dim)
				}
			}
			// concat: [head0row0]...[head0rowN][head1row0]... => [head0row0][head1row0]...[headNrow0]...
			// size: (inps, head_dim * num_heads) => (inps, d_mod)
#pragma omp parallel
			{
				size_t inf_head_size = posd * head_dim;
#pragma omp for collapse(2)
				for (size_t head = 0; head < num_heads; head++) {
					for (size_t i = 0; i < inputs_size; i++) {
						const float* restrict h_o = &l_o[head * inf_head_size];
						float* restrict h_concat = &l_concat[head * head_dim];
						const float* restrict h_o_row = &h_o[i * head_dim];
						float* restrict h_concat_row = &h_concat[i * d_mod];
						memcpy(h_concat_row, h_o_row, head_dim * sizeof(*concat));
					}
				}
				// concat(inps, d_mod) @ W_o(d_mod, d_mod) = attention(inps, d_mod)
				matmul(l_concat, l_w_o, l_out, inputs_size, d_mod, d_mod);

				// Residual Attention
#pragma omp for simd
				for (size_t idx = 0; idx < inputs_size * d_mod; idx++) {
					l_res[idx] = layer_input[idx] + l_out[idx];
				}
				// LayerNorm2
				{
					const float* restrict gamma = &gamma2[jump_param];
					const float* restrict beta = &beta2[jump_param];
					float* restrict l_mean = &mean2[jump_inference];
					float* restrict l_inv_std = &inv_std2[jump_inference];
					float* restrict l_x_hat = &x_hat2[jump_inference];
					float* restrict l_ln = &ln2[jump_inference];
					layernorm_inf_train(
						inputs_size, d_mod, d,
						eps, l_res, gamma,
						beta, l_mean, l_inv_std,
						l_x_hat, l_ln
					);
					// x = (inps, d_mod)
				}
				// FeedForward
				{
					const float* restrict l_w_ffn1 = &w_ffn1[jump_param]; // (d_mod, ffd)
					const float* restrict l_w_ffn2 = &w_ffn2[jump_param]; // (ffd, d_mod)
					const float* restrict l_ln_inp = &ln2[jump_inference];
					float* restrict l_ff_pre_act = &ff_pre_act[jump_inference];
					float* restrict l_ff_out1 = &ff_out1[jump_inference];
					float* restrict l_ff_out2 = &ff_out2[jump_inference];

					//memset(l_ff_pre_act, 0, inputs_size * ffd * sizeof(*l_ff_pre_act));
					//memset(l_ff_out2, 0, inputs_size * d_mod * sizeof(*l_ff_out2));
#pragma omp for simd
					for (size_t i = 0; i < inputs_size * ffd; i++) {
						l_ff_pre_act[i] = 0;
					}
#pragma omp for simd
					for (size_t i = 0; i < inputs_size * d_mod; i++) {
						l_ff_out2[i] = 0;
					}

					// (inps, d_mod) @ (d_mod, ffd) => (inps, ffd)
					matmul(l_ln_inp, l_w_ffn1, l_ff_pre_act, inputs_size, d_mod, ffd);
#pragma omp for simd
					for (size_t idx = 0; idx < inputs_size * ffd; idx++) {
						const float x = l_ff_pre_act[idx];
						l_ff_out1[idx] = x / (1.0f + expf(-x));
					}
					// (inps, ffd) @ (ffd, d_mod) => (inps, d_mod)
					matmul(l_ff_out1, l_w_ffn2, l_ff_out2, inputs_size, ffd, d_mod);
					// Residual FeedForward(inps, d_mod)
#pragma omp for simd
					for (size_t idx = 0; idx < inputs_size * d_mod; idx++) {
						l_ff_res[idx] = l_res[idx] + l_ff_out2[idx];
					}
				}
			}
		}
		layer_input = l_ff_res;
	}
	// LM Head
	memset(logits, 0, inputs_size * vocd * sizeof(*logits));
#pragma omp parallel
	{
		matmul(layer_input, w_lmhead, logits, inputs_size, d_mod, vocd);

#pragma omp for
		for (size_t i = 0; i < inputs_size; i++) {
			float* restrict logits_row = &logits[i * vocd];
			float max = -INFINITY;

			// max
#pragma omp simd
			for (size_t k = 0; k < vocd; k++) {
				const float x = logits_row[k];
				if (x > max) max = x;
			}

			// softmax
			float sum = 0.0f;
#pragma omp simd reduction(+:sum)
			for (size_t k = 0; k < vocd; k++) {
				const float x = expf(logits_row[k] - max);
				logits_row[k] = x;
				sum += x;
			}

			sum = (1.0f / sum);
#pragma omp simd
			for (size_t k = 0; k < vocd; k++) {
				logits_row[k] *= sum;
			}
		}
	}
	/*
	 * if label == 0 then skip
	 * else if label == 1 then train
	 */

	float loss = 0.0f; 
	size_t numL = 0;
#pragma omp parallel
	{
		// Calculate loss
// 1. Compute loss (parallel reduction over tokens)
#pragma omp for reduction(+:loss, numL)
		for (size_t i = 0; i < inputs_size; i++) {
			if (labels[i]) {
				const float* restrict probs = &logits[i * vocd];
				size_t t = target[i];
				float p = probs[t];
				loss += -logf(p > 1e-12f ? p : 1e-12f);
				numL++;
			} 
		}

		float grad_factor = (numL > 0) ? (1.0f / (float)numL) : 0.0f;

		// 2. Compute gradients (parallel over all (i,j) pairs) dlogits
#pragma omp for 
		for (size_t i = 0; i < inputs_size; i++) {
			float* restrict dlog_row = &dlogits[i * vocd];
			if (labels[i]) {
				const float* restrict log_row = &logits[i * vocd];
				const size_t t = target[i];

#pragma omp simd
				for (size_t j = 0; j < vocd; j++) {
					float grad = log_row[j];
					dlog_row[j] = grad * grad_factor;
				}
				{
					float grad = log_row[t] - 1.0f;
					dlog_row[t] = grad * grad_factor;
				}
			} else {
#pragma omp simd
				for (size_t j = 0; j < vocd; j++) {
					dlog_row[j] = 0.0f;
				}
			}
		}

		// LM Head
		// Calc dW_lm[
		// layer_input^T = (posd, d_mod), 
		// dlogits = (posd, vocd),
		// grad_lm_head = (d_mod, vocd)
		// ]
		matmulT_(layer_input, dlogits, grad_lm_head, inputs_size, d_mod, vocd);
		// Calc dY
		memset(dY, 0, inputs_size * d_mod * sizeof(*dY));
		matmul_T(dlogits, w_lmhead, dY, inputs_size, vocd, d_mod);
	}

	// Layer backward
	for (size_t layer = num_layers; layer-- > 0; ) {
		size_t jump_inference = layer * layer_inference_size;
		size_t jump_param = layer * layer_size;
#pragma omp parallel
		{
			// FeedForward
			{
				// weights
				const float* restrict l_w_ffn1 = &w_ffn1[jump_param]; // (d_mod, ffd)
				const float* restrict l_w_ffn2 = &w_ffn2[jump_param]; // (ffd, d_mod)
				// workspace
				const float* restrict l_ln_inp = &ln2[jump_inference];
				float* restrict l_ff_pre_act = &ff_pre_act[jump_inference];
				float* restrict l_ff_out1 = &ff_out1[jump_inference]; // (inps, ffd)
				// grads
				float* restrict l_grad_wffn2 = &grad_W_ffn2[jump_param];

				// feedforward2
				matmulT_(l_ff_out1, dY, l_grad_wffn2, inputs_size, ffd, d_mod);

#pragma omp for
				for (size_t i = 0; i < inputs_size; i++) {
					const float* restrict _pre_act_row = &l_ff_pre_act[i * ffd];
					float* restrict _out_row = &dffn1[i * ffd];
					const float* restrict a_row = &dY[i * d_mod];
					for (size_t k = 0; k < ffd; k++) {
						const float* restrict b_row = &l_w_ffn2[k * d_mod];

						float act = _pre_act_row[k];

						float sum = 0.0f;
#pragma omp simd reduction(+:sum)
						for (size_t j = 0; j < d_mod; j++) {
							sum += a_row[j] * b_row[j];
						}

						float act_sig = (1.0f / (1.0f + expf(-act)));
						float der_sig = act_sig * (1.0f - act_sig);
						float silu = act_sig + act * der_sig;
						_out_row[k] = sum * silu;
					}
				}

				// feedforward1
				// dW1 += LayerNorm2^T @ dFF_pre_act  (dFF_pre_act is already in dFF_out_1)
				matmulT_(l_ln_inp, dffn1, grad_W_ffn1, inputs_size, d_mod, ffd);

				// dLn2 = dffn1 @ W_ffn1^T
				memset(dlayernorm, 0, inputs_size * d_mod * sizeof(*dlayernorm));
				matmul_T(dffn1, l_w_ffn1, dlayernorm, inputs_size, ffd, d_mod);

				// layernorm + residual #2(add to dY)
				{
					const float* restrict gamma = &gamma2[jump_param];
					float* restrict l_grad_gamma = &grad_gamma2[jump_param];
					float* restrict l_grad_beta = &grad_beta2[jump_param];
					float* restrict l_inv_std = &inv_std2[jump_inference];
					float* restrict l_x_hat = &x_hat2[jump_inference];

					layernorm_grad(
						dlayernorm, l_x_hat, l_inv_std, 
						gamma, dY, thread_gamma, 
						thread_beta, l_grad_gamma, l_grad_beta, 
						inputs_size, d_mod
					);
				}

				// concat
				{
					const float* restrict l_concat = &concat[jump_inference];
					const float* restrict l_W_o = &W_o[jump_param];
					float* restrict l_grad_W_o = &grad_W_o[jump_param];

					// dW_o   += concat^T @ dY
					matmulT_(l_concat, dY, l_grad_W_o, inputs_size, d_mod, d_mod);

					// dConcat = dY @ W_o^T
					memset(dconcat, 0, inputs_size * d_mod * sizeof(*dconcat));
					matmul_T(dY, l_W_o, dconcat, inputs_size, d_mod, d_mod);
				}
			}
		}
		// MHA
		{
			// inputs
			const float* restrict l_inp = &ln1[jump_inference];
			// weights
			const float* restrict l_w_q = &W_q[jump_param];
			const float* restrict l_w_k = &W_k[jump_param];
			const float* restrict l_w_v = &W_v[jump_param];
			// buffers
			const float* restrict l_q = &q[jump_inference];
			const float* restrict l_k = &k[jump_inference];
			const float* restrict l_v = &v[jump_inference];
			// grads
			float* restrict l_g_w_q = &grad_W_q[jump_param];
			float* restrict l_g_w_k = &grad_W_k[jump_param];
			float* restrict l_g_w_v = &grad_W_v[jump_param];
			
			memset(dlayernorm, 0, inputs_size * d_mod * sizeof(*dlayernorm));


			for (size_t head = 0; head < num_heads; head++) {
				size_t idx_w = head * head_size; //[head0][head1]...[headn]
				size_t idx_f = head * posd * head_dim; //[head0][head1]...[headn]
				const float* restrict h_w_q = &l_w_q[idx_w];
				const float* restrict h_w_k = &l_w_k[idx_w];
				const float* restrict h_w_v = &l_w_v[idx_w];
				float* restrict h_g_w_q = &l_g_w_q[idx_w];
				float* restrict h_g_w_k = &l_g_w_k[idx_w];
				float* restrict h_g_w_v = &l_g_w_v[idx_w];
				const float* restrict h_q = &l_q[idx_f];
				const float* restrict h_k = &l_k[idx_f];
				const float* restrict h_v = &l_v[idx_f];

				memset(datt, 0, inputs_size* inputs_size * sizeof(*datt));
				memset(dV, 0, inputs_size* head_dim * sizeof(*datt));
				memset(dK, 0, inputs_size* head_dim * sizeof(*datt));
				memset(dQ, 0, inputs_size* head_dim * sizeof(*datt));


#pragma omp parallel
				{
					// dO_head
					{
						// [row0head0][row0head1]->[row0head0][row1head0], size = (inps, d_mod)
#pragma omp for
						for (size_t i = 0; i < inputs_size; i++) {
							size_t src = i * d_mod + head * head_dim;
							size_t dst = i * head_dim;
							memcpy(&dO_head[dst], &dconcat[src], head_dim * sizeof(*dO_head));
						}
					}

					// att = softmax(QK^T / √d)
					Scaled_Dot_Product_Attention(h_q, h_k, att, inputs_size, head_dim, inputs_size, 0); // (inps, inps)

					// dV = att^T @ dO_head 
					matmulT_(att, dO_head, dV, inputs_size, inputs_size, head_dim);

					// datt = dO_head(inps, head_dim) @ V^T(inps, head_dim)
					matmul_T(dO_head, h_v, datt, inputs_size, head_dim, inputs_size); // (inps, inps)

					// dAtt = (dAtt * Att - Att * rowsum(dAtt * Att)) * (1 / sqrt(attention_size))
					const float scale = 1 / sqrtf((float)head_dim);
#pragma omp for
					for (size_t i = 0; i < inputs_size; i++) {
						float* restrict dAtt_row = &datt[i * inputs_size];
						const float* restrict Att_row = &att[i * inputs_size];
						float sum = 0.0f;
						// TODO: Refactor the loop to calculate the boundary, 
						// in order to eliminate unnecessary cycles 
						// and avoid processing "zero-any" cases.
#pragma omp simd reduction(+:sum)
						for (size_t j = 0; j < inputs_size; j++) {
							const float x = dAtt_row[j] * Att_row[j];
							dAtt_row[j] = x;
							sum += x;
						}

#pragma omp simd
						for (size_t j = 0; j < inputs_size; j++) {
							float x = dAtt_row[j] - Att_row[j] * sum;
							dAtt_row[j] = x * scale;
						}
					}

					// dQ = dAtt @ K
					matmul(datt, h_k, dQ, inputs_size, inputs_size, head_dim);

					// dK = dAtt^T @ Q
					matmulT_(datt, h_q, dK, inputs_size, inputs_size, head_dim);

					// dW_q_h += ln1^T @ dQ       [inps, d_mod]^T @ [inps, head_dim] = [d_mod, head_dim]
					matmulT_(l_inp, dQ, h_g_w_q, inputs_size, d_mod, head_dim);

					// dW_k_h += ln^T @ dK
					matmulT_(l_inp, dK, h_g_w_k, inputs_size, d_mod, head_dim);

					// dW_v_h += ln^T @ dV
					matmulT_(l_inp, dV, h_g_w_v, inputs_size, d_mod, head_dim);

					// dlayernorm += dQ @ W_q_h^T (inps, head_dim) @ (d_mod, head_dim)^T => (inps, d_mod)
					matmul_T(dQ, h_w_q, dlayernorm, inputs_size, head_dim, d_mod);

					// dlayernorm += dK @ W_k_h^T
					matmul_T(dK, h_w_k, dlayernorm, inputs_size, head_dim, d_mod);

					// dlayernorm += dV @ W_v_h^T
					matmul_T(dV, h_w_v, dlayernorm, inputs_size, head_dim, d_mod);
				}
			}

			// layernorm1 + residual #1
#pragma omp parallel
			{
				const float* restrict gamma = &gamma1[jump_param];
				float* restrict l_grad_gamma = &grad_gamma1[jump_param];
				float* restrict l_grad_beta = &grad_beta1[jump_param];
				float* restrict l_inv_std = &inv_std1[jump_inference];
				float* restrict l_x_hat = &x_hat1[jump_inference];

				layernorm_grad(
					dlayernorm, l_x_hat, l_inv_std,
					gamma, dY, thread_gamma,
					thread_beta, l_grad_gamma, l_grad_beta,
					inputs_size, d_mod
				);
			}
		}
	}

#pragma omp parallel
	{
		// Embedding and Position
#pragma omp for
		for (size_t i = 0; i < inputs_size; i++) {
			const float* enc_row = &dY[i * d_mod];
			float* emb_row = &grad_embedding[inputs[i] * d_mod];
			float* pos_row = &grad_positional[i * d_mod];
			// positional
#pragma omp simd
			for (size_t j = 0; j < d_mod; j++) {
				pos_row[j] += enc_row[j];
			}
			// embedding
			for (size_t j = 0; j < d_mod; j++) {
#pragma omp atomic
				emb_row[j] += enc_row[j];
			}
			
		}
	}

	workspace->batchs++;

	loss /= (float)numL;

	return loss;
}

float OptimizerStep(
	Model* self,
	const float learning_rate,
	const float beta1,
	const float beta2,
	const float eps,
	const float clip_norm,
	const float weight_decay
) {
	if (!self) return 0.0f;
	if (
		!self->config ||
		!self->weights ||
		!self->workspace ||
		!self->adamw
		) return 0.0f;

	Workspace* workspace = self->workspace;
	Weights* weights = self->weights;
	AdamW* adamw = self->adamw;

	float* restrict data = weights->data;
	float* restrict grads = weights->grad;
	float* restrict rmsprop = adamw->rmsprop;
	float* restrict momentum = adamw->momentum;

	adamw->step++;
	const size_t t = adamw->step;
	const size_t size = weights->size_model;
	const float scale = 1.0f / (float)workspace->batchs;

	// bias correction
	const float bias_correction1 = 1.0f / (1 - powf(beta1, (float)t));
	const float bias_correction2 = 1.0f / (1 - powf(beta2, (float)t));

	float global_norm = 0.0f;
#pragma omp parallel for simd reduction(+:global_norm)
	for (size_t i = 0; i < size; i++) {
		const float g = grads[i] * scale;
		grads[i] = g;
		global_norm += g * g;
	}
	global_norm = sqrtf(global_norm);

	float clip_scale = 1.0f;
	if (clip_norm > 0.0f && global_norm > clip_norm) clip_scale = clip_norm / global_norm;

#pragma omp parallel for simd
	for (size_t i = 0; i < size; i++) {
		const float g = grads[i] * clip_scale;
		grads[i] = 0.0f;
		// momentum = momentum * beta1 + (1 - beta1) * grad
		const float moment = momentum[i] * beta1 + (1 - beta1) * g;
		momentum[i] = moment;
		// rmsprop = rmsprop * beta2 + (1 - beta2) * grad**2
		const float rmsprp = rmsprop[i] * beta2 + (1 - beta2) * (g * g);
		rmsprop[i] = rmsprp;

		// m_hat = momentum/(1 - beta1**t)
		const float m_hat = moment * bias_correction1;
		// v_hat = rmsprop/(1 - beta2**t)
		const float v_hat = rmsprp * bias_correction2;

		// update = lr * (m_hat / (sqrt(v_hat) + eps) + wd * x[i]
		const float update = learning_rate * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * data[i]);

		// weight = weight - update
		data[i] -= update;
	}

	workspace->batchs = 0;

	return global_norm;
}
#endif


void AttachTokenizer(
	Model* model, 
	Tokenizer* tokenizer
) {
	if (!model || !tokenizer) {
		fprintf(stderr, "AttachTokenizer: NULL argument\n");
		return;
	}

	if (model->tokenizer && model->tokenizer != tokenizer) {
		fprintf(stderr, "AttachTokenizer: replacing existing tokenizer\n");
		tok_destroy(model->tokenizer);
	}

	model->tokenizer = tokenizer;
}

/*
 * All multi-byte values are little-endian.
 * file structure:
 * 1. File header identifying the file (magic string) -> (char)
 * 2. File version -> (uint32_t)
 * 3. A flag indicating whether to save training gradients. -> (uint8_t)
 * 4. model_size (required for integrity check) -> (uint64_t)
 * 5. Model configuration: 
 *         - vocab_size -> (uint64_t)
 *         - position_size -> (uint64_t)
 *         - d_model -> (uint64_t)
 *         - num_heads -> (uint64_t)
 *         - num_layers -> (uint64_t)
 *         - ffn_dim -> (uint64_t)
 *         - eps -> (float)
 * 6. Model weights -> (float)
 * if with gradients:
 *     7. Model gradients -> (float)
 *     8. Number of optimizer steps  -> (uint64_t)
 *     9. momentum for AdamW -> (float)
 *     10. rmsprop for AdamW -> (float)
 *     11. Model tokenizer -> (See the configuration in the tokenizer files)
 * else:
 *     7. Model tokenizer -> (See the configuration in the tokenizer files)
 */
int SaveModel(const Model* model, const int with_gradients, const char* filepath) {
	if (!model || !filepath) {
		fprintf(stderr, "A NULL pointer was received in one of the arguments.\n");
		return 1;
	}
	if (with_gradients != 0 && with_gradients != 1) {
		fprintf(stderr, "Invalid with_gradients flag: %d\n", with_gradients);
		return 1;
	}
	if (!model->config || !model->weights || !model->tokenizer) {
		fprintf(stderr, "Failed to retrieve the Config, Tokenizer or Weights structure.\n");
		return 1;
	}
	if (!model->weights->data) {
		fprintf(stderr, "Failed to retrieve the data array.\n");
		return 1;
	}
#if MOUEN_TRAINING
	if (with_gradients && !model->adamw) {
		fprintf(stderr, "Failed to retrieve the AdamW structure.\n");
		return 1;
	}
	if (with_gradients && (!model->weights->grad || !model->adamw->momentum || !model->adamw->rmsprop)) {
		fprintf(stderr, "Failed to retrieve grad, momentum or rmsprop array.\n");
		return 1;
	}
#endif 
	FILE* file = fopen(filepath, "wb");
	if (!file) {
		fprintf(stderr, "%s\n", strerror(errno));
		return 1;
	}

	Config* config = model->config;
	Weights* weights = model->weights;

	char magic[] = MOUEN_MAGIC;
#if MOUEN_TRAINING
	uint8_t with_grad = (uint8_t)with_gradients;
#else
	uint8_t with_grad = (uint8_t)0;
#endif
	uint32_t version = MOUEN_FILE_VERSION;
	uint64_t size_t_to_uint64_t;

	if (fwrite(magic, sizeof(magic), 1, file) != 1) {
		fprintf(stderr, "Error writing header\n");
		fclose(file);
		return 1;
	}

	if (fwrite(&version, sizeof(version), 1, file) != 1) {
		fprintf(stderr, "Error writing version\n");
		fclose(file);
		return 1;
	}

	if (fwrite(&with_grad, sizeof(with_grad), 1, file) != 1) {
		fprintf(stderr, "Error writing with_gradients\n");
		fclose(file);
		return 1;
	}

	size_t_to_uint64_t = (uint64_t)weights->size_model;
	if (fwrite(&size_t_to_uint64_t, sizeof(size_t_to_uint64_t), 1, file) != 1) {
		fprintf(stderr, "Error writing size_model\n");
		fclose(file);
		return 1;
	}

	size_t_to_uint64_t = (uint64_t)config->vocab_size;
	if (fwrite(&size_t_to_uint64_t, sizeof(size_t_to_uint64_t), 1, file) != 1) {
		fprintf(stderr, "Error writing vocab_size\n");
		fclose(file);
		return 1;
	}

	size_t_to_uint64_t = (uint64_t)config->position_size;
	if (fwrite(&size_t_to_uint64_t, sizeof(size_t_to_uint64_t), 1, file) != 1) {
		fprintf(stderr, "Error writing position_size\n");
		fclose(file);
		return 1;
	}

	size_t_to_uint64_t = (uint64_t)config->d_model;
	if (fwrite(&size_t_to_uint64_t, sizeof(size_t_to_uint64_t), 1, file) != 1) {
		fprintf(stderr, "Error writing d_model\n");
		fclose(file);
		return 1;
	}

	size_t_to_uint64_t = (uint64_t)config->num_heads;
	if (fwrite(&size_t_to_uint64_t, sizeof(size_t_to_uint64_t), 1, file) != 1) {
		fprintf(stderr, "Error writing num_heads\n");
		fclose(file);
		return 1;
	}	
	
	size_t_to_uint64_t = (uint64_t)config->num_layers;
	if (fwrite(&size_t_to_uint64_t, sizeof(size_t_to_uint64_t), 1, file) != 1) {
		fprintf(stderr, "Error writing num_layers\n");
		fclose(file);
		return 1;
	}

	size_t_to_uint64_t = (uint64_t)config->ffn_dim;
	if (fwrite(&size_t_to_uint64_t, sizeof(size_t_to_uint64_t), 1, file) != 1) {
		fprintf(stderr, "Error writing ffn_dim\n");
		fclose(file);
		return 1;
	}

	if (fwrite(&config->eps, sizeof(config->eps), 1, file) != 1) {
		fprintf(stderr, "Error writing eps\n");
		fclose(file);
		return 1;
	}

	size_t size_model = weights->size_model;
	if (fwrite(weights->data, sizeof(*weights->data), size_model, file) != size_model) {
		fprintf(stderr, "Error writing data\n");
		fclose(file);
		return 1;
	}

#if MOUEN_TRAINING
	if (with_gradients) {
		AdamW* adamw = model->adamw;

		if (fwrite(weights->grad, sizeof(*weights->grad), size_model, file) != size_model) {
			fprintf(stderr, "Error writing grad\n");
			fclose(file);
			return 1;
		}

		size_t_to_uint64_t = (uint64_t)adamw->step;
		if (fwrite(&size_t_to_uint64_t, sizeof(size_t_to_uint64_t), 1, file) != 1) {
			fprintf(stderr, "Error writing step\n");
			fclose(file);
			return 1;
		}

		if (fwrite(adamw->momentum, sizeof(*adamw->momentum), size_model, file) != size_model) {
			fprintf(stderr, "Error writing momentum\n");
			fclose(file);
			return 1;
		}

		if (fwrite(adamw->rmsprop, sizeof(*adamw->rmsprop), size_model, file) != size_model) {
			fprintf(stderr, "Error writing rmsprop\n");
			fclose(file);
			return 1;
		}
	}
#endif

	if (tok_save(model->tokenizer, file) != 0) {
		fprintf(stderr, "Error writing tokenizer\n");
		fclose(file);
		return 1;
	}

	if (fclose(file) != 0) {
		fprintf(stderr, "Error closing file: %s\n", strerror(errno));
		return 1;
	}

	return 0;
}

static inline int fread_size_param(
	size_t* out,
	const char* desc,
	FILE* file
) {
	uint64_t x;
	if (fread(&x, sizeof(x), 1, file) != 1) {
		fprintf(stderr, "Error reading %s\n", desc);
		return 1;
	}
#if SIZE_MAX < UINT64_MAX
	if (x > SIZE_MAX) {
		fprintf(stderr, "%s too large\n", desc);
		return 1;
	}
#endif
	* out = (size_t)x;
	return 0;
}

Model* LoadModel(const char* filepath, const int use_mmap) {
	if (!filepath || use_mmap < 0 || use_mmap > 1) return NULL;

	FILE* file = fopen(filepath, "rb");
	if (!file) {
		fprintf(stderr, "%s\n", strerror(errno));
		return NULL;
	}

	{
		char magic[sizeof(MOUEN_MAGIC)];
		if (fread(magic, sizeof(*magic), sizeof(magic), file) != sizeof(magic)) {
			fprintf(stderr, "Error reading header\n");
			fclose(file);
			return NULL;
		}

		if (memcmp(magic, MOUEN_MAGIC, sizeof(MOUEN_MAGIC)) != 0) {
			fprintf(stderr, "Not a Mouen file\n");
			fclose(file);
			return NULL;
		}
	}

	uint32_t version;
	{
		if (fread(&version, sizeof(version), 1, file) != 1) {
			fprintf(stderr, "Error reading version\n");
			fclose(file);
			return NULL;
		}

		if (version > MOUEN_FILE_VERSION) {
			fprintf(stderr, "Unsupported file version: %u (max %u)\n", version, MOUEN_FILE_VERSION);
			fclose(file);
			return NULL;
		}
	}

	uint8_t with_gradients;
	{
		if (fread(&with_gradients, sizeof(with_gradients), 1, file) != 1) {
			fprintf(stderr, "Error reading with_gradients\n");
			fclose(file);
			return NULL;
		}

		if (with_gradients != 0 && with_gradients != 1) {
			fprintf(stderr, "Invalid with_gradients flag: %u\n", with_gradients);
			fclose(file);
			return NULL;
		}

#if !MOUEN_TRAINING
		if (with_gradients) {
			fprintf(stderr, "It is not possible to load a training file into an inference-only build.\n");
			fclose(file);
			return NULL;
		}
#endif // !MOUEN_TRAINING

	}

#ifdef _WIN32
	if (use_mmap) {
		fprintf(stderr, "Warning: mmap not supported on Windows, using malloc\n");
	}
	const int use_mmap_now = 0;
#else
	const int use_mmap_now = (with_gradients) ? 0 : use_mmap;
#endif

	size_t model_size;
	if (fread_size_param(&model_size, "model_size", file)) { fclose(file); return NULL;	}

	Model* model = NULL;
	{
		size_t vocab_size, position_size, d_model, num_heads, num_layers, ffn_dim;

		if (fread_size_param(&vocab_size, "vocab_size", file)) { fclose(file); return NULL; }
		if (fread_size_param(&position_size, "position_size", file)) { fclose(file); return NULL; }
		if (fread_size_param(&d_model, "d_model", file)) { fclose(file); return NULL; }
		if (fread_size_param(&num_heads, "num_heads", file)) { fclose(file); return NULL; }
		if (fread_size_param(&num_layers, "num_layers", file)) { fclose(file); return NULL; }
		if (fread_size_param(&ffn_dim, "ffn_dim", file)) { fclose(file); return NULL; }

		float eps;
		if (fread(&eps, sizeof(eps), 1, file) != 1) {
			fprintf(stderr, "Error reading eps\n");
			fclose(file);
			return NULL;
		}

		model = CreateModel(
			vocab_size, position_size, d_model,
			num_heads, num_layers, ffn_dim,
			eps
		);
		if(!model) {
			fprintf(stderr, "Error create Model\n");
			fclose(file);
			return NULL;
		}

		if (model_size != model->weights->size_model) {
			fprintf(stderr, "size_model mismatch: file=%zu, config=%zu\n",
				model_size, model->weights->size_model);
			FreeModel(&model);
			fclose(file);
			return NULL;
		}
	}

	{
		Config* config = model->config;
		Weights* weights = model->weights;

		// import settings
		size_t d_mod = config->d_model;
		size_t vocd = config->vocab_size;
		size_t posd = config->position_size;
		size_t ffd = config->ffn_dim;
		size_t size_model = weights->size_model;

		if (use_mmap_now) {
#ifndef _WIN32
			// we get the header size
			off_t header_size = ftello(file);
			if (header_size < 0) {
				fprintf(stderr, "ftello failed: %s\n", strerror(errno));
				FreeModel(&model);
				fclose(file);
				return NULL;
			}

			// handle to mmap
			int fd = fileno(file);

			// We request memory equal to the size of the weights plus the size of the header.
			size_t bytes_of_weights = model_size * sizeof(*weights->data);
			size_t total_size = (size_t)header_size + bytes_of_weights;


			void* mapped_ptr = mmap(NULL, total_size, PROT_READ, MAP_PRIVATE, fd, 0);
			if (mapped_ptr == MAP_FAILED) {
				fprintf(stderr, "Failed to mapping for the data array.\n");
				FreeModel(&model);
				fclose(file);
				return NULL;
			}

			if (fseeko(file, (off_t)bytes_of_weights, SEEK_CUR) != 0) {
				fprintf(stderr, "fseeko failed: %s\n", strerror(errno));
				munmap(mapped_ptr, total_size);
				FreeModel(&model);
				fclose(file);
				return NULL;
			}

			weights->mmap_ptr = mapped_ptr;
			weights->size_mmap = total_size;
			weights->data = (float*)((char*)mapped_ptr + header_size);
			weights->is_mmap = 1;


#endif
		}
		else {
			weights->data = calloc(model_size, sizeof(*weights->data));
			if (!weights->data) {
				fprintf(stderr, "Failed to allocate memory for the data array.\n");
				FreeModel(&model);
				fclose(file);
				return NULL;
			}

			if (fread(weights->data, sizeof(*weights->data), model_size, file) != model_size) {
				fprintf(stderr, "Error reading weights\n");
				FreeModel(&model);
				fclose(file);
				return NULL;
			}

			weights->is_mmap = 0;
			weights->size_mmap = 0;
			weights->mmap_ptr = NULL;
		}

		// layout of array pointers in memory
		{
			size_t bias = 0;
			// embedding
			weights->embedding = &weights->data[bias];
			bias += vocd * d_mod;
			// positional
			weights->positional = &weights->data[bias];
			bias += posd * d_mod;
			// layer
			for (size_t layer = 0; layer < config->num_layers; layer++) {
				// gamma1
				if (layer == 0) weights->gamma1 = &weights->data[bias];
				bias += d_mod;
				// beta1
				if (layer == 0) weights->beta1 = &weights->data[bias];
				bias += d_mod;
				// w_q
				if (layer == 0) weights->W_q = &weights->data[bias];
				bias += d_mod * d_mod;
				// w_k
				if (layer == 0) weights->W_k = &weights->data[bias];
				bias += d_mod * d_mod;
				// w_v
				if (layer == 0) weights->W_v = &weights->data[bias];
				bias += d_mod * d_mod;
				// w_o
				if (layer == 0) weights->W_o = &weights->data[bias];
				bias += d_mod * d_mod;
				// gamma2
				if (layer == 0) weights->gamma2 = &weights->data[bias];
				bias += d_mod;
				// beta2
				if (layer == 0) weights->beta2 = &weights->data[bias];
				bias += d_mod;
				// ffn1
				if (layer == 0) weights->W_ffn1 = &weights->data[bias];
				bias += d_mod * ffd;
				// ffn2
				if (layer == 0) weights->W_ffn2 = &weights->data[bias];
				bias += d_mod * ffd;
			}
			// LM Head
			weights->lm_head = &weights->data[bias];
			bias += d_mod * vocd;

			if (bias != size_model) {
				fprintf(stderr, "layout mismatch: bias=%zu, size_model=%zu\n", bias, size_model);
				FreeModel(&model);
				fclose(file);
				return NULL;
			}
		}
	}

#if MOUEN_TRAINING
	if (with_gradients) {
		if (InitTrain(model)) {
			fprintf(stderr, "Failed to allocate memory for the training arrays.\n");
			FreeModel(&model);
			fclose(file);
			return NULL;
		}

		if (fread(model->weights->grad, sizeof(*model->weights->grad), model_size, file) != model_size) {
			fprintf(stderr, "Error reading grad\n");
			FreeModel(&model);
			fclose(file);
			return NULL;
		}

		if (fread_size_param(&model->adamw->step, "step", file)) {
			fprintf(stderr, "Error reading step\n");
			FreeModel(&model);
			fclose(file);
			return NULL;
		}

		if (fread(model->adamw->momentum, sizeof(*model->adamw->momentum), model_size, file) != model_size) {
			fprintf(stderr, "Error reading momentum\n");
			FreeModel(&model);
			fclose(file);
			return NULL;
		}

		if (fread(model->adamw->rmsprop, sizeof(*model->adamw->rmsprop), model_size, file) != model_size) {
			fprintf(stderr, "Error reading rmsprop\n");
			FreeModel(&model);
			fclose(file);
			return NULL;
		}
	}
#endif
	// tokenizer
	{
		Tokenizer* tokenizer = tok_load(file);
		if (!tokenizer) {
			FreeModel(&model);
			fclose(file);
			return NULL;
		}

		AttachTokenizer(model, tokenizer);
	}


	if (fclose(file) != 0) {
		fprintf(stderr, "Error closing file: %s\n", strerror(errno));
		FreeModel(&model);
		return NULL;
	}

	return model;
}