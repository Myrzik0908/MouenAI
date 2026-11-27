#include "tokenizer.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <errno.h>
#include <omp.h>

// For use with Windows
#ifdef _WIN32
#include <windows.h>
#endif 


/*
 * structure:
 * 1. num_tokens -> (uint64_t)
 * 2. lengths of all tokens -> (uint64_t)
 * 3. vocab -> (char)
 */
static inline GreedyData* tok_greedy_load(FILE* file) {
	if (!file) return NULL;

	// create a tokenizer object
	GreedyData* self = calloc(1, sizeof(*self));
	if (!self) {
		fprintf(stderr, "Error create tokenizer\n");
		return NULL;
	}

	// Variable for comparisons
	uint64_t num_tokens;
	if (fread(&num_tokens, sizeof(num_tokens), 1, file) != 1) {
		fprintf(stderr, "Error reading parameter num_tokens\n");
		free(self);
		return NULL;
	}

	// size_t overflow check:
	if (num_tokens > SIZE_MAX) {
		fprintf(stderr, "num_tokens too large\n");
		free(self);
		return NULL;
	}

	self->num_tokens = (size_t)num_tokens;

	// allocate memory for the dictionary array
	self->size_token = calloc(self->num_tokens, sizeof(*self->size_token));
	if (!self->size_token) {
		fprintf(stderr, "Error allocating memory\n");
		free(self);
		return NULL;
	}

	{
		uint64_t* tmp = NULL;
		if (self->num_tokens > SIZE_MAX / sizeof(*tmp)) {
			fprintf(stderr, "Too many tokens\n");
			free(self->size_token);
			free(self);
			return NULL;
		}

		// allocate memory for the temp array
		tmp = malloc(self->num_tokens * sizeof(*tmp));
		if (!tmp) {
			fprintf(stderr, "Error allocating memory\n");
			free(self);
			return NULL;
		}

		// load len of tokens array
		if (fread(tmp, sizeof(*tmp), self->num_tokens, file) != self->num_tokens) {
			fprintf(stderr, "Error reading parameter len of tokens\n");
			free(self->size_token);
			free(self);
			free(tmp);
			return NULL;
		}

#pragma omp parallel for simd
		for (size_t i = 0; i < self->num_tokens; i++) {
			self->size_token[i] = (size_t)tmp[i];
		}

		free(tmp);
	}

	// allocate memory for the offsets array
	self->offsets = calloc(self->num_tokens, sizeof(*self->offsets));
	if (!self->offsets) {
		fprintf(stderr, "Error allocating memory for offsets\n");
		free(self->size_token);
		free(self);
		return NULL;
	}

	// we find the total length of the array
	// Python: vocab + '\0' IMPORTANT
	size_t summary = 1;
	for (size_t i = 0; i < self->num_tokens; i++) {
		summary += self->size_token[i];
	}

	// allocate memory for the dictionary array
	self->vocab = calloc(summary, sizeof(*self->vocab));
	if (!self->vocab) {
		fprintf(stderr, "Error allocating memory for vocab\n");
		free(self->offsets);
		free(self->size_token);
		free(self);
		return NULL;
	}

	// load vocab
	if (fread(self->vocab, sizeof(*self->vocab), summary, file) != summary) {
		fprintf(stderr, "Error reading parameter vocab\n");
		free(self->offsets);
		free(self->size_token);
		free(self);
		return NULL;
	}

	// calculating offsets
	size_t pos = 0;
	for (size_t i = 0; i < self->num_tokens; i++) {
		self->offsets[i] = pos;
		pos += self->size_token[i];
	}

	return self;
}

int tok_import_tokenizer_from_bin(Tokenizer* self, const char* filepath) {
	if (!self || !filepath) return -1;

	FILE* file = fopen(filepath, "rb");
	if (!file) {
		fprintf(stderr, "%s\n", strerror(errno));
		return -1;
	}

	TokenizerType type = self->type;

	switch (type) {
	case TOK_GREEDY:
		self->data = tok_greedy_load(file);
		if (!self->data) {
			fclose(file);
			return -1;
		}
		break;
	default:
		fprintf(stderr, "Invalid tokenizer type\n");
		fclose(file);
		return -1;
	}

	if (fclose(file) != 0) {
		fprintf(stderr, "Error closing file: %s\n", strerror(errno));
		return -1;
	}

	return 0;
}

Tokenizer* tok_create(TokenizerType type, const char* filepath) {
#ifdef _WIN32
	SetConsoleOutputCP(CP_UTF8);
	SetConsoleCP(CP_UTF8);
#endif
	// Protection against non-existent types
	switch (type) {
	case TOK_GREEDY:
		break;
	default:
		fprintf(stderr, "Invalid tokenizer type\n");
		return NULL;
	}

	Tokenizer* self = calloc(1, sizeof(*self));
	if (!self) {
		fprintf(stderr, "Can't create object\n");
		return NULL;
	}

	self->type = type;

	if (filepath) {
		if (tok_import_tokenizer_from_bin(self, filepath) != 0) {
			free(self);
			return NULL;
		}
	}

	return self;
}

static inline void tok_greedy_printf(const GreedyData* self) {
	if (!self) return;
	size_t len = self->num_tokens;
	char* vocab = self->vocab;
	size_t* lens = self->size_token;
	size_t* offsets = self->offsets;
	printf("num tokens: %zu\nVocab:\n", len + 1); // 'cause + unk
	printf("idx: %zu, token(System-level, encoded automatically!): %s\n", (size_t)0, "unknown");
	for (size_t i = 0; i < len; i++) {
		size_t current_len = lens[i];
		size_t offset = offsets[i];
		printf("len token: %zu, idx: %zu, token: %.*s\n", current_len, i + 1, (int)current_len, &vocab[offset]);
		//printf("%.*s\n", (int)current_len, &vocab[offset]);
	}
	fflush(stdout);
}

void tok_printf(const Tokenizer* self) {
	if (!self) {
		fprintf(stderr, "No tokenizer\n");
		return;
	}

	if (!self->data) {
		fprintf(stderr, "Tokenizer not loaded\n");
		return;
	}

	switch (self->type) {
	case TOK_GREEDY:
		tok_greedy_printf((GreedyData*)self->data);
		return;
	default:
		fprintf(stderr, "Invalid tokenizer type\n");
		return;
	}
}

static inline void tok_greedy_destroy(GreedyData* self) {
	if (!self) return;
	free(self->offsets);
	free(self->vocab);
	free(self->size_token);
	free(self);
}

void tok_destroy(Tokenizer* self) {
	if (!self) {
		fprintf(stderr, "No tokenizer\n");
		return;
	}


	switch (self->type) {
	case TOK_GREEDY:
		tok_greedy_destroy((GreedyData*)self->data);
		break;
	default:
		fprintf(stderr, "Invalid tokenizer type\n");
		break;
	}
	free(self);
}

size_t tok_num_tokens(const Tokenizer* self) {
	if (!self) {
		fprintf(stderr, "No tokenizer\n");
		return 0;
	}

	if (!self->data) {
		fprintf(stderr, "Tokenizer not loaded\n");
		return 0;
	}

	switch (self->type) {
	case TOK_GREEDY:
		return ((GreedyData*)self->data)->num_tokens + 1; // N + unk
	default:
		fprintf(stderr, "Invalid tokenizer type\n");
		return 0;
	}
}

static inline size_t greedy_get_token_for_seq(const GreedyData* obj, const char* seq) {
	// IMPORTANT: vocab must be sorted by length descending.
	// return error
	if (!obj || !seq) return 0;

	/*
	 * struct:
	 * 0 index - return error;
	 * 1 index - return unknown token;
	 * 2...n index - return token from vocab;
	 */

	// Getting param
	const size_t num_tokens = obj->num_tokens;
	const size_t* offsets = obj->offsets;
	const size_t* tokens_len = obj->size_token;
	const char* vocab = obj->vocab;
	for (size_t idx = 0; idx < num_tokens; idx++) {
		const char* token = &vocab[offsets[idx]];
		const size_t len = tokens_len[idx];
		int equal = 1;
		for (size_t i = 0; i < len; i++) {
			const char now_char = seq[i];
			if (now_char == '\0' || token[i] != now_char) {
				equal = 0;
				break;
			}
		}
		if (equal) {
			return idx + 2; // 'cause 0-err, 1-unk, 2n-vocab
		}
	}

	// return unknown token
	return 1;
}

static inline size_t tok_greedy_encode(const GreedyData* self, const char* sequense, size_t* buffer, const size_t len_buffer) {
	if (!self || !sequense || !buffer) return 0;

	const size_t* len_tokens = self->size_token;

	size_t offset = 0;
	size_t write = 0;
	while (sequense[offset] != '\0') {
		if (write >= len_buffer) return 0;   // buffer full

		size_t token = greedy_get_token_for_seq(self, &sequense[offset]);

		// unknown
		if (token == 1) {
			buffer[write] = 0;
			offset++;
		}
		else {
			buffer[write] = token - 1;
			offset += len_tokens[token - 2];
		}
		write++;
	}
	return write;
}

size_t tok_encode(const Tokenizer* self, const char* sequense, size_t* buffer, const size_t len_buffer) {
	if (!self) {
		fprintf(stderr, "No tokenizer\n");
		return 0;
	}

	if (len_buffer == 0) return 0;

	if (!self->data) {
		fprintf(stderr, "Tokenizer not loaded\n");
		return 0;
	}

	switch (self->type) {
	case TOK_GREEDY:
		return tok_greedy_encode((GreedyData*)self->data, sequense, buffer, len_buffer);
	default:
		fprintf(stderr, "Invalid tokenizer type\n");
		return 0;
	}
}

/*void tok_decode(const Tokenizer* self, const size_t* sequense, char* buffer, const size_t len_sequense, const size_t len_buffer) {
	if (!self || !sequense || !buffer || len_sequense < 1) return;

	const size_t* offsets = self->offsets;
	const char* vocab = self->vocab;
	const size_t* len_tokens = self->size_token;
	const size_t num_tokens = self->num_tokens;
	size_t pos = 0;
	for (size_t i = 0; i < len_sequense; i++) {
		size_t token = sequense[i];
		if (token == 0 || token > num_tokens) continue;
		token--;
		const size_t len = len_tokens[token];
		if (pos + len + 1 > len_buffer) return;
		memcpy(&buffer[pos], &vocab[offsets[token]], len * sizeof(char));
		pos += len;
	}
	buffer[pos] = '\0';
}*/

static int tok_greedy_save(const GreedyData* self, FILE* file) {
	if (!self || !file) return 1;

	const size_t num_tokens = self->num_tokens;
	const uint64_t uint64_t_num_tokens = (uint64_t)num_tokens;
	const size_t* tokens_len = self->size_token;
	const char* vocab = self->vocab;

	// write number of tokens
	if (fwrite(&uint64_t_num_tokens, sizeof(uint64_t_num_tokens), 1, file) != 1) {
		fprintf(stderr, "Error writing parameter num_tokens\n");
		return 1;
	}

	// write lengths
	{
		uint64_t* tmp = NULL;
		if (num_tokens > SIZE_MAX / sizeof(*tmp)) {
			fprintf(stderr, "Too many tokens\n");
			return 1;
		}

		// allocate memory for the temp array
		tmp = malloc(num_tokens * sizeof(*tmp));
		if (!tmp) {
			fprintf(stderr, "Error allocating memory\n");
			return 1;
		}

#pragma omp parallel for simd
		for (size_t i = 0; i < num_tokens; i++) {
			tmp[i] = (uint64_t)tokens_len[i];
		}

		if (fwrite(tmp, sizeof(*tmp), num_tokens, file) != num_tokens) {
			fprintf(stderr, "Error writing parameter len of tokens\n");
			free(tmp);
			return 1;
		}

		free(tmp);
	}

	size_t summary = 1;
	for (size_t i = 0; i < num_tokens; i++) {
		size_t len = tokens_len[i];
		summary += len;
	}

	if (fwrite(vocab, sizeof(*vocab), summary, file) != summary) {
		fprintf(stderr, "Error writing parameter vocab\n");
		return 1;
	}

	return 0;
}

/*
 * Saves the tokenizer to an open file.
 * Layout:
 *   [type: uint32_t]
 *   [tokenizer-specific data]
 * self - tokenizer object (must not be NULL, data must be loaded)
 * file - open file handle (must not be NULL)
 * Returns: 0 on success, 1 on error
 */
int tok_save(const Tokenizer* self, FILE* file) {
	if (!self || !file) return 1;

	if (!self->data) {
		fprintf(stderr, "Tokenizer not loaded\n");
		return 1;
	}

	// Type validation:
	switch (self->type) {
	case TOK_GREEDY: break;
	default:
		fprintf(stderr, "Invalid tokenizer type\n");
		return 1;
	}

	uint32_t type_ser = (uint32_t)self->type;
	if (fwrite(&type_ser, sizeof(type_ser), 1, file) != 1) {
		fprintf(stderr, "Error writing tokenizer type\n");
		return 1;
	}

	switch (self->type) {
	case TOK_GREEDY:
		return tok_greedy_save((const GreedyData*)self->data, file);
	}

	return 1;   // unreachable
}

Tokenizer* tok_load(FILE* file) {
	if (!file) return NULL;

	uint32_t type;
	if (fread(&type, sizeof(type), 1, file) != 1) {
		fprintf(stderr, "Error reading type_tokenizer\n");
		return NULL;
	}

	Tokenizer* tokenizer = tok_create((TokenizerType)type, NULL);
	if (!tokenizer) {
		fprintf(stderr, "Error create tokenizer\n");
		return NULL;
	}

	switch ((TokenizerType)type) {
	case TOK_GREEDY: 
		tokenizer->data = tok_greedy_load(file);
		if (!tokenizer->data) {
			tok_destroy(tokenizer);
			return NULL;
		}
		break;
	default:
		fprintf(stderr, "Invalid tokenizer type\n");
		tok_destroy(tokenizer);
		return NULL;
	}

	return tokenizer;
}

static void greedyprintf(const GreedyData* self, const size_t token) {
	if (!self) return;
	size_t len = self->num_tokens;
	char* vocab = self->vocab;
	size_t* lens = self->size_token;
	size_t* offsets = self->offsets;

	if (token == 0 || token > len) return;

	const size_t idx = token - 1;

	printf("%.*s", (int)lens[idx], &vocab[offsets[idx]]);
	fflush(stdout);
}

void tok_tokenprintf(Tokenizer* self, const size_t token) {
	if (!self) {
		fprintf(stderr, "No tokenizer\n");
		return;
	}

	if (!self->data) {
		fprintf(stderr, "Tokenizer not loaded\n");
		return;
	}

	switch (self->type) {
	case TOK_GREEDY:
		greedyprintf((const GreedyData*)self->data, token);
		break;
	default:
		fprintf(stderr, "Invalid tokenizer type\n");
		break;
	}
}