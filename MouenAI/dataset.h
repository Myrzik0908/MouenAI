#pragma once
#include <stdio.h>
#include <stdint.h>

typedef struct {
	size_t* tokens;
	uint8_t* labels;
	size_t num_tokens;
} Dataset;
Dataset* DatasetLoad(const char* filepath);

void DatasetDetails(Dataset* self);

void DatasetFree(Dataset* self);

size_t GetRandomIdx(Dataset* self, const size_t context);