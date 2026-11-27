#include "dataset.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>


Dataset* DatasetLoad(const char* filepath) {
	if (!filepath) return NULL;

	Dataset* self = malloc(sizeof(*self));
	if (!self) return NULL;

	FILE* file = fopen(filepath, "rb");
	if (!file) {
		printf("%s\n", strerror(errno));
		free(self);
		return NULL;
	}

	if (fread(&self->num_tokens, sizeof(self->num_tokens), 1, file) != 1) {
		printf("Error reading data and loading into num_tokens!\n");
		free(self);
		fclose(file);
		return NULL;
	}

	self->tokens = calloc(self->num_tokens, sizeof(*self->tokens));
	if (!self->tokens) {
		printf("Failed to allocate memory for data!\n");
		free(self);
		fclose(file);
		return NULL;
	}

	if (fread(self->tokens, sizeof(*self->tokens), self->num_tokens, file) != self->num_tokens) {
		printf("Error reading data and loading into tokens!\n");
		free(self->tokens);
		free(self);
		fclose(file);
		return NULL;
	}

	self->labels = calloc(self->num_tokens, sizeof(*self->labels));
	if (!self->labels) {
		printf("Failed to allocate memory for labels!\n");
		free(self->tokens);
		free(self);
		fclose(file);
		return NULL;
	}

	if (fread(self->labels, sizeof(*self->labels), self->num_tokens, file) != self->num_tokens) {
		printf("Error reading data and loading into tokens!\n");
		free(self->tokens);
		free(self);
		fclose(file);
		return NULL;
	}

	fclose(file);

	return self;
}

void DatasetDetails(Dataset* self) {
	if (!self) return;
	size_t max = (self->num_tokens < 100) ? self->num_tokens : 100;
	for (size_t i = 0; i < max; i++) printf("%zu->%hhu ", self->tokens[i], self->labels[i]);
	printf("len dataset: %zu\n", self->num_tokens);
}

void DatasetFree(Dataset* self) {
	if (!self) return;
	free(self->labels);
	free(self->tokens);
	free(self);
}


size_t GetRandomIdx(Dataset* self, const size_t context) {
	if (!self || self->num_tokens <= context) return 0;

	size_t num_tokens = self->num_tokens;

	size_t max_valid_idx = num_tokens - context;

	unsigned int big_rand = ((unsigned int)rand() << 15) | (rand() & 0x7FFF);

	size_t idx = (size_t)(big_rand % (max_valid_idx + 1));

	return idx;
}