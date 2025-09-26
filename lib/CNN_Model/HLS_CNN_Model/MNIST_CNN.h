// parameters.h
#ifndef PARAMETERS_H
#define PARAMETERS_H

#include "ap_fixed.h"

// Data types
typedef ap_fixed<16, 6> input_t;     // 16-bit fixed point, 6 integer bits
typedef ap_fixed<16, 6> weight_t;    // Same for weights
typedef ap_fixed<32, 12> result_t;   // Wider for accumulation

// Alternative: use float for simplicity
// typedef float input_t;
// typedef float weight_t;
// typedef float result_t;

#endif

// MNIST_CNN.h
#ifndef MNIST_CNN_H
#define MNIST_CNN_H

#include "parameters.h"

// Function declarations
void MNIST_CNN(input_t* input, input_t* weights, input_t* output);

// Utility functions
void input_reshape(input_t* input, input_t OFM[1][28][28]);
void output_reshape(input_t IFM[10], input_t* output);

// Padding functions
void padding_input(input_t IFM[1][28][28], input_t OFM[1][30][30]);
void padding_3ch(input_t IFM[3][14][14], input_t OFM[3][16][16]);
void padding_9ch(input_t IFM[9][7][7], input_t OFM[9][9][9]);

// Weight reshape functions
void wreshape_conv1(input_t* w, input_t WBUF[3][1][3][3]);
void wreshape_conv2(input_t* w, input_t WBUF[9][3][3][3]);
void wreshape_conv3(input_t* w, input_t WBUF[9][9][3][3]);

// Convolution layers
void conv1_layer(input_t IFM[1][30][30], input_t WBUF[3][1][3][3], input_t OFM[3][28][28], input_t* bias);
void conv2_layer(input_t IFM[3][16][16], input_t WBUF[9][3][3][3], input_t OFM[9][14][14], input_t* bias);
void conv3_layer(input_t IFM[9][9][9], input_t WBUF[9][9][3][3], input_t OFM[9][7][7], input_t* bias);

// Activation functions
void relu_3ch(input_t IFM[3][28][28], input_t OFM[3][28][28]);
void relu_9ch_14(input_t IFM[9][14][14], input_t OFM[9][14][14]);
void relu_9ch_7(input_t IFM[9][7][7], input_t OFM[9][7][7]);

// Pooling functions
void maxpool_3ch(input_t IFM[3][28][28], input_t OFM[3][14][14]);
void maxpool_9ch_14(input_t IFM[9][14][14], input_t OFM[9][7][7]);
void maxpool_9ch_7(input_t IFM[9][7][7], input_t OFM[9][3][3]);

// Flatten and fully connected
void flatten(input_t IFM[9][3][3], input_t OFM[81]);
void fc1_layer(input_t IFM[81], input_t* weight, input_t* bias, input_t OFM[128]);
void fc2_layer(input_t IFM[128], input_t* weight, input_t* bias, input_t OFM[64]);
void fc3_layer(input_t IFM[64], input_t* weight, input_t* bias, input_t OFM[10]);

#endif

// Example weight file structure (weights/conv1_weight.h)
#ifndef CONV1_WEIGHT_H
#define CONV1_WEIGHT_H

#include "../parameters.h"

// Conv1: 3 output channels, 1 input channel, 3x3 kernel = 27 weights
extern const input_t conv1_weight[27];

#endif

// Example bias file structure (weights/conv1_bias.h)
#ifndef CONV1_BIAS_H
#define CONV1_BIAS_H

#include "../parameters.h"

// Conv1: 3 output channels = 3 biases
extern const input_t conv1_bias[3];

#endif

// Example Makefile for HLS synthesis
/*
# Makefile for MNIST CNN HLS

PROJECT = mnist_cnn
TOP_FUNCTION = MNIST_CNN
SOLUTION = solution1
DEVICE = xc7z020