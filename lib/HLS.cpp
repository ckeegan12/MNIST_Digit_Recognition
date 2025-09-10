#include <iostream>
using namespace std;

// Data types for MNIST CNN
typedef float input_t;
typedef float weight_t;
typedef float result_t;

// Operation sizes
#define IMG_SIZE 28 // Width and height 
#define PADDED_SIZE 30 // With and height with padding=1
#define CONV1_OUT 28
#define POOL1_OUT 14 // 28 / (pooling_size=2)
#define CONV2_OUT 14
#define POOL2_OUT 7 // 14 / (pooling_size=2)
#define FC_INPUT 81 // 9*3*3  
#define KERNEL 3 // 3x3 Kernel

// Network parameters
#define CONV1_IN_CH 1
#define CONV1_OUT_CH 3
#define CONV2_IN_CH 3
#define CONV2_OUT_CH 9
#define CONV3_IN_CH 9
#define CONV3_OUT_CH 9
#define FC1_UNITS 128
#define FC2_UNITS 64
#define NUM_CLASSES 10 // images of digits

// Weight arrays
weight_t conv1_weights[CONV1_IN_CH][CONV1_OUT_CH][KERNEL][KERNEL];
weight_t conv1_bias[CONV1_OUT];
weight_t conv2_weights[CONV2_FILTERS][CONV1_FILTERS][3][3];
weight_t conv2_bias[CONV2_FILTERS];
weight_t fc1_weights[FC_UNITS][FC_INPUT];
weight_t fc1_bias[FC_UNITS];
weight_t fc2_weights[NUM_CLASSES][FC_UNITS];
weight_t fc2_bias[NUM_CLASSES];

// Internal buffers (replace streams)
input_t padded_img[PADDED_SIZE][PADDED_SIZE];
input_t conv1_out[CONV1_FILTERS][CONV1_OUT][CONV1_OUT];
input_t pool1_out[CONV1_FILTERS][POOL1_OUT][POOL1_OUT];
input_t padded_conv1[CONV1_FILTERS][POOL1_OUT+2][POOL1_OUT+2];
input_t conv2_out[CONV2_FILTERS][CONV2_OUT][CONV2_OUT];
input_t pool2_out[CONV2_FILTERS][POOL2_OUT][POOL2_OUT];
input_t fc1_out[FC_UNITS];
input_t flattened[FC_INPUT];