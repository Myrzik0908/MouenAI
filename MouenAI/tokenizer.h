#pragma once
#include <stdio.h>
#include <stdlib.h>

typedef enum : int {
	TOK_GREEDY = 0,
} TokenizerType;

typedef struct {
	char* vocab; // load
	size_t* size_token; // load
	size_t* offsets; // automatically
	size_t num_tokens; // load
} GreedyData;

typedef struct {
	void* data;
	TokenizerType type;
} Tokenizer;

/*
 * Creates a tokenizer object with the specified algorithm.
 * type - type of tokenizer used
 * filepath - an optional parameter that loads a separately prepared .bin file for the tokenizer
 */
Tokenizer* tok_create(TokenizerType type, const char* filepath);

/*
 * Outputs a dictionary of all tokens.
 * self - a dictionary object.
 */
void tok_printf(const Tokenizer* self);

/*
 * Removes the dictionary from memory.
 * self - a dictionary object.
 */
void tok_destroy(Tokenizer* self);

/*
 * Encodes a text sequence into a sequence of tokens.
 * self - a dictionary object.
 * sequense - input text for tokenizer processing
 * buffer - buffer where the result will be written
 * len_buffer - buffer length for protection against out-of-bounds access
 */
size_t tok_encode(const Tokenizer* self, const char* sequense, size_t* buffer, const size_t len_buffer);

//void tok_decode(const Tokenizer* self, const size_t* sequense, char* buffer, const size_t len_sequense, const size_t len_buffer);


/*
 * Prints the token to the console using printf
 * self - a dictionary object.
 * token - printable token
 */
void tok_tokenprintf(Tokenizer* self, const size_t token);

/*
 * Saves the tokenizer to a file.
 * self - tokenizer object (must not be NULL)
 * file - open file handle to write into (must not be NULL)
 * IMPORTANT: does not close the file in case of an error
 * Returns: 0 on success, 1 on error
 */
int tok_save(const Tokenizer* self, FILE* file);

/*
 *
 * file - open file handle to read (must not be NULL)
 * IMPORTANT: does not close the file in case of an error
 * Returns: pointer on success, NULL on error
 */
Tokenizer* tok_load(FILE* file);

/*
 * Returns the number of tokens in the dictionary.
 * self - a dictionary object.
 */
size_t tok_num_tokens(const Tokenizer* self);

/*
 *
 * self - a dictionary object.
 */
int tok_import_tokenizer_from_bin(Tokenizer* self, const char* filepath);
