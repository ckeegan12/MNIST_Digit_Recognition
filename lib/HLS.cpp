#include <iostream>
#include <ap_fixed.h>
#include <hls_math.h>
#include <parameters.h>
using namespace std;

// Define fixed-point data types for hardware efficiency
typedef ap_fixed<16, 8> data_t;
typedef ap_fixed<16, 8> weight_t;
typedef ap_fixed<32, 16> acc_t;

// Network parameters based on your PyTorch model
#define INPUT_HEIGHT 28
#define INPUT_WIDTH 28
#define INPUT_CHANNELS 1

// Layer dimensions
#define CONV1_OUT_CH 3
#define CONV2_OUT_CH 9
#define CONV3_OUT_CH 9
#define FC1_OUT 128
#define FC2_OUT 64
#define FC3_OUT 10

// Kernel size
#define KERNEL_SIZE 3
#define PADDING 1

// Include weight headers (you'll need to generate these from your PyTorch weights)
#include "weights/conv1_weight.h"
#include "weights/conv1_bias.h"
#include "weights/conv2_weight.h"
#include "weights/conv2_bias.h"
#include "weights/conv3_weight.h"
#include "weights/conv3_bias.h"
#include "weights/fc1_weight.h"
#include "weights/fc1_bias.h"
#include "weights/fc2_weight.h"
#include "weights/fc2_bias.h"
#include "weights/fc3_weight.h"
#include "weights/fc3_bias.h"