#include <iostream>
#include "parameters.h"
#include "MNIST_CNN.h"
using namespace std;
#include <fstream>

// Weight includes
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

// Layer feature maps
input_t INPUT[1][28][28];
input_t CONV1_OUT[3][28][28];
input_t POOL1_OUT[3][14][14];
input_t CONV2_OUT[9][14][14];
input_t POOL2_OUT[9][7][7];
input_t CONV3_OUT[9][7][7];
input_t POOL3_OUT[9][3][3];

// Padded feature maps
input_t PADDED_INPUT[1][30][30];
input_t PADDED_CONV1[3][16][16];
input_t PADDED_CONV2[9][9][9];

// Weight buffers
input_t WBUF_CONV1[3][1][3][3];
input_t WBUF_CONV2[9][3][3][3];
input_t WBUF_CONV3[9][9][3][3];

// Flattened and FC buffers
input_t FLATTENED[81];  // 9 * 3 * 3 = 81
input_t FC1_OUT[128];
input_t FC2_OUT[64];
input_t FC3_OUT[10];

// Utility functions
void padding_input(input_t IFM[1][28][28], input_t OFM[1][30][30]){
    #pragma HLS ARRAY_PARTITION variable=IFM complete dim=1
    #pragma HLS ARRAY_PARTITION variable=OFM complete dim=1
    
    for(int i = 0; i < 1; i++){
        for(int j = 0; j < 28; j++){
            for(int k = 0; k < 28; k++){
                #pragma HLS PIPELINE
                OFM[i][j+1][k+1] = IFM[i][j][k];
            }
        }
    }
    
    // Zero padding
    for(int i = 0; i < 1; i++){
        for(int j = 0; j < 30; j++){
            OFM[i][j][0] = 0;
            OFM[i][j][29] = 0;
        }
        for(int k = 0; k < 30; k++){
            OFM[i][0][k] = 0;
            OFM[i][29][k] = 0;
        }
    }
}

void padding_3ch(input_t IFM[3][14][14], input_t OFM[3][16][16]){
    #pragma HLS ARRAY_PARTITION variable=IFM complete dim=1
    #pragma HLS ARRAY_PARTITION variable=OFM complete dim=1
    
    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 14; j++){
            for(int k = 0; k < 14; k++){
                #pragma HLS PIPELINE
                OFM[i][j+1][k+1] = IFM[i][j][k];
            }
        }
    }
    
    // Zero padding
    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 16; j++){
            OFM[i][j][0] = 0;
            OFM[i][j][15] = 0;
        }
        for(int k = 0; k < 16; k++){
            OFM[i][0][k] = 0;
            OFM[i][15][k] = 0;
        }
    }
}

void padding_9ch(input_t IFM[9][7][7], input_t OFM[9][9][9]){
    #pragma HLS ARRAY_PARTITION variable=IFM complete dim=1
    #pragma HLS ARRAY_PARTITION variable=OFM complete dim=1
    
    for(int i = 0; i < 9; i++){
        for(int j = 0; j < 7; j++){
            for(int k = 0; k < 7; k++){
                #pragma HLS PIPELINE
                OFM[i][j+1][k+1] = IFM[i][j][k];
            }
        }
    }
    
    // Zero padding
    for(int i = 0; i < 9; i++){
        for(int j = 0; j < 9; j++){
            OFM[i][j][0] = 0;
            OFM[i][j][8] = 0;
        }
        for(int k = 0; k < 9; k++){
            OFM[i][0][k] = 0;
            OFM[i][8][k] = 0;
        }
    }
}

// Weight reshape functions
void wreshape_conv1(input_t* w, input_t WBUF[3][1][3][3]){
    #pragma HLS ARRAY_PARTITION variable=WBUF complete dim=1
    #pragma HLS ARRAY_PARTITION variable=WBUF complete dim=2
    
    for(int oc = 0; oc < 3; oc++){
        for(int ic = 0; ic < 1; ic++){
            for(int kh = 0; kh < 3; kh++){
                for(int kw = 0; kw < 3; kw++){
                    #pragma HLS PIPELINE
                    WBUF[oc][ic][kh][kw] = w[oc*1*3*3 + ic*3*3 + kh*3 + kw];
                }
            }
        }
    }
}

void wreshape_conv2(input_t* w, input_t WBUF[9][3][3][3]){
    #pragma HLS ARRAY_PARTITION variable=WBUF complete dim=1
    #pragma HLS ARRAY_PARTITION variable=WBUF complete dim=2
    
    for(int oc = 0; oc < 9; oc++){
        for(int ic = 0; ic < 3; ic++){
            for(int kh = 0; kh < 3; kh++){
                for(int kw = 0; kw < 3; kw++){
                    #pragma HLS PIPELINE
                    WBUF[oc][ic][kh][kw] = w[oc*3*3*3 + ic*3*3 + kh*3 + kw];
                }
            }
        }
    }
}

void wreshape_conv3(input_t* w, input_t WBUF[9][9][3][3]){
    #pragma HLS ARRAY_PARTITION variable=WBUF complete dim=1
    #pragma HLS ARRAY_PARTITION variable=WBUF complete dim=2
    
    for(int oc = 0; oc < 9; oc++){
        for(int ic = 0; ic < 9; ic++){
            for(int kh = 0; kh < 3; kh++){
                for(int kw = 0; kw < 3; kw++){
                    #pragma HLS PIPELINE
                    WBUF[oc][ic][kh][kw] = w[oc*9*3*3 + ic*3*3 + kh*3 + kw];
                }
            }
        }
    }
}

// Convolution layers
void conv1_layer(input_t IFM[1][30][30], input_t WBUF[3][1][3][3], input_t OFM[3][28][28], input_t* bias){
    #pragma HLS ARRAY_PARTITION variable=IFM complete dim=1
    #pragma HLS ARRAY_PARTITION variable=WBUF complete dim=1
    #pragma HLS ARRAY_PARTITION variable=WBUF complete dim=2
    #pragma HLS ARRAY_PARTITION variable=OFM complete dim=1
    
    for(int oc = 0; oc < 3; oc++){
        for(int oh = 0; oh < 28; oh++){
            for(int ow = 0; ow < 28; ow++){
                #pragma HLS PIPELINE
                input_t sum = 0;
                for(int ic = 0; ic < 1; ic++){
                    for(int kh = 0; kh < 3; kh++){
                        for(int kw = 0; kw < 3; kw++){
                            #pragma HLS UNROLL
                            sum += IFM[ic][oh+kh][ow+kw] * WBUF[oc][ic][kh][kw];
                        }
                    }
                }
                OFM[oc][oh][ow] = sum + bias[oc];
            }
        }
    }
}

void conv2_layer(input_t IFM[3][16][16], input_t WBUF[9][3][3][3], input_t OFM[9][14][14], input_t* bias){
    #pragma HLS ARRAY_PARTITION variable=IFM complete dim=1
    #pragma HLS ARRAY_PARTITION variable=WBUF complete dim=1
    #pragma HLS ARRAY_PARTITION variable=WBUF complete dim=2
    #pragma HLS ARRAY_PARTITION variable=OFM complete dim=1
    
    for(int oc = 0; oc < 9; oc++){
        for(int oh = 0; oh < 14; oh++){
            for(int ow = 0; ow < 14; ow++){
                #pragma HLS PIPELINE
                input_t sum = 0;
                for(int ic = 0; ic < 3; ic++){
                    for(int kh = 0; kh < 3; kh++){
                        for(int kw = 0; kw < 3; kw++){
                            #pragma HLS UNROLL
                            sum += IFM[ic][oh+kh][ow+kw] * WBUF[oc][ic][kh][kw];
                        }
                    }
                }
                OFM[oc][oh][ow] = sum + bias[oc];
            }
        }
    }
}

void conv3_layer(input_t IFM[9][9][9], input_t WBUF[9][9][3][3], input_t OFM[9][7][7], input_t* bias){
    #pragma HLS ARRAY_PARTITION variable=IFM complete dim=1
    #pragma HLS ARRAY_PARTITION variable=WBUF complete dim=1
    #pragma HLS ARRAY_PARTITION variable=WBUF complete dim=2
    #pragma HLS ARRAY_PARTITION variable=OFM complete dim=1
    
    for(int oc = 0; oc < 9; oc++){
        for(int oh = 0; oh < 7; oh++){
            for(int ow = 0; ow < 7; ow++){
                #pragma HLS PIPELINE
                input_t sum = 0;
                for(int ic = 0; ic < 9; ic++){
                    for(int kh = 0; kh < 3; kh++){
                        for(int kw = 0; kw < 3; kw++){
                            #pragma HLS UNROLL
                            sum += IFM[ic][oh+kh][ow+kw] * WBUF[oc][ic][kh][kw];
                        }
                    }
                }
                OFM[oc][oh][ow] = sum + bias[oc];
            }
        }
    }
}

// ReLU activation
void relu_3ch(input_t IFM[3][28][28], input_t OFM[3][28][28]){
    #pragma HLS ARRAY_PARTITION variable=IFM complete dim=1
    #pragma HLS ARRAY_PARTITION variable=OFM complete dim=1
    
    for(int c = 0; c < 3; c++){
        for(int h = 0; h < 28; h++){
            for(int w = 0; w < 28; w++){
                #pragma HLS PIPELINE
                OFM[c][h][w] = (IFM[c][h][w] > 0) ? IFM[c][h][w] : 0;
            }
        }
    }
}

void relu_9ch_14(input_t IFM[9][14][14], input_t OFM[9][14][14]){
    #pragma HLS ARRAY_PARTITION variable=IFM complete dim=1
    #pragma HLS ARRAY_PARTITION variable=OFM complete dim=1
    
    for(int c = 0; c < 9; c++){
        for(int h = 0; h < 14; h++){
            for(int w = 0; w < 14; w++){
                #pragma HLS PIPELINE
                OFM[c][h][w] = (IFM[c][h][w] > 0) ? IFM[c][h][w] : 0;
            }
        }
    }
}

void relu_9ch_7(input_t IFM[9][7][7], input_t OFM[9][7][7]){
    #pragma HLS ARRAY_PARTITION variable=IFM complete dim=1
    #pragma HLS ARRAY_PARTITION variable=OFM complete dim=1
    
    for(int c = 0; c < 9; c++){
        for(int h = 0; h < 7; h++){
            for(int w = 0; w < 7; w++){
                #pragma HLS PIPELINE
                OFM[c][h][w] = (IFM[c][h][w] > 0) ? IFM[c][h][w] : 0;
            }
        }
    }
}

// Max pooling layers
void maxpool_3ch(input_t IFM[3][28][28], input_t OFM[3][14][14]){
    #pragma HLS ARRAY_PARTITION variable=IFM complete dim=1
    #pragma HLS ARRAY_PARTITION variable=OFM complete dim=1
    
    for(int c = 0; c < 3; c++){
        for(int h = 0; h < 14; h++){
            for(int w = 0; w < 14; w++){
                #pragma HLS PIPELINE
                input_t max_val = IFM[c][h*2][w*2];
                input_t val1 = IFM[c][h*2][w*2+1];
                input_t val2 = IFM[c][h*2+1][w*2];
                input_t val3 = IFM[c][h*2+1][w*2+1];
                
                if(val1 > max_val) max_val = val1;
                if(val2 > max_val) max_val = val2;
                if(val3 > max_val) max_val = val3;
                
                OFM[c][h][w] = max_val;
            }
        }
    }
}

void maxpool_9ch_14(input_t IFM[9][14][14], input_t OFM[9][7][7]){
    #pragma HLS ARRAY_PARTITION variable=IFM complete dim=1
    #pragma HLS ARRAY_PARTITION variable=OFM complete dim=1
    
    for(int c = 0; c < 9; c++){
        for(int h = 0; h < 7; h++){
            for(int w = 0; w < 7; w++){
                #pragma HLS PIPELINE
                input_t max_val = IFM[c][h*2][w*2];
                input_t val1 = IFM[c][h*2][w*2+1];
                input_t val2 = IFM[c][h*2+1][w*2];
                input_t val3 = IFM[c][h*2+1][w*2+1];
                
                if(val1 > max_val) max_val = val1;
                if(val2 > max_val) max_val = val2;
                if(val3 > max_val) max_val = val3;
                
                OFM[c][h][w] = max_val;
            }
        }
    }
}

void maxpool_9ch_7(input_t IFM[9][7][7], input_t OFM[9][3][3]){
    #pragma HLS ARRAY_PARTITION variable=IFM complete dim=1
    #pragma HLS ARRAY_PARTITION variable=OFM complete dim=1
    
    for(int c = 0; c < 9; c++){
        for(int h = 0; h < 3; h++){
            for(int w = 0; w < 3; w++){
                #pragma HLS PIPELINE
                input_t max_val = IFM[c][h*2][w*2];
                if(h*2+1 < 7 && w*2+1 < 7){
                    input_t val1 = IFM[c][h*2][w*2+1];
                    input_t val2 = IFM[c][h*2+1][w*2];
                    input_t val3 = IFM[c][h*2+1][w*2+1];
                    
                    if(val1 > max_val) max_val = val1;
                    if(val2 > max_val) max_val = val2;
                    if(val3 > max_val) max_val = val3;
                }
                OFM[c][h][w] = max_val;
            }
        }
    }
}

// Flatten layer
void flatten(input_t IFM[9][3][3], input_t OFM[81]){
    #pragma HLS ARRAY_PARTITION variable=OFM complete
    
    int idx = 0;
    for(int c = 0; c < 9; c++){
        for(int h = 0; h < 3; h++){
            for(int w = 0; w < 3; w++){
                #pragma HLS PIPELINE
                OFM[idx++] = IFM[c][h][w];
            }
        }
    }
}

// Fully connected layers
void fc1_layer(input_t IFM[81], input_t* weight, input_t* bias, input_t OFM[128]){
    #pragma HLS ARRAY_PARTITION variable=IFM complete
    #pragma HLS ARRAY_PARTITION variable=OFM complete
    
    for(int oc = 0; oc < 128; oc++){
        #pragma HLS PIPELINE
        input_t sum = 0;
        for(int ic = 0; ic < 81; ic++){
            #pragma HLS UNROLL factor=8
            sum += IFM[ic] * weight[oc*81 + ic];
        }
        input_t result = sum + bias[oc];
        OFM[oc] = (result > 0) ? result : 0;  // ReLU
    }
}

void fc2_layer(input_t IFM[128], input_t* weight, input_t* bias, input_t OFM[64]){
    #pragma HLS ARRAY_PARTITION variable=IFM complete
    #pragma HLS ARRAY_PARTITION variable=OFM complete
    
    for(int oc = 0; oc < 64; oc++){
        #pragma HLS PIPELINE
        input_t sum = 0;
        for(int ic = 0; ic < 128; ic++){
            #pragma HLS UNROLL factor=8
            sum += IFM[ic] * weight[oc*128 + ic];
        }
        input_t result = sum + bias[oc];
        OFM[oc] = (result > 0) ? result : 0;  // ReLU
    }
}

void fc3_layer(input_t IFM[64], input_t* weight, input_t* bias, input_t OFM[10]){
    #pragma HLS ARRAY_PARTITION variable=IFM complete
    #pragma HLS ARRAY_PARTITION variable=OFM complete
    
    for(int oc = 0; oc < 10; oc++){
        #pragma HLS PIPELINE
        input_t sum = 0;
        for(int ic = 0; ic < 64; ic++){
            #pragma HLS UNROLL
            sum += IFM[ic] * weight[oc*64 + ic];
        }
        OFM[oc] = sum + bias[oc];  // No activation for final layer
    }
}

// Input reshape
void input_reshape(input_t* input, input_t OFM[1][28][28]){
    for(int h = 0; h < 28; h++){
        for(int w = 0; w < 28; w++){
            #pragma HLS PIPELINE
            OFM[0][h][w] = input[h*28 + w];
        }
    }
}

// Output reshape
void output_reshape(input_t IFM[10], input_t* output){
    for(int i = 0; i < 10; i++){
        #pragma HLS PIPELINE
        output[i] = IFM[i];
    }
}

// Main function
void MNIST_CNN(input_t* input, input_t* weights, input_t* output){
    #pragma HLS INTERFACE m_axi depth=784 port=input offset=slave bundle=input
    #pragma HLS INTERFACE m_axi depth=50000 port=weights offset=slave bundle=weights
    #pragma HLS INTERFACE m_axi depth=10 port=output offset=slave bundle=output
    #pragma HLS INTERFACE s_axilite port=return
    
    // Resource allocation constraints
    #pragma HLS ALLOCATION function instances=padding_input limit=1
    #pragma HLS ALLOCATION function instances=padding_3ch limit=1
    #pragma HLS ALLOCATION function instances=padding_9ch limit=1
    #pragma HLS ALLOCATION function instances=conv1_layer limit=1
    #pragma HLS ALLOCATION function instances=conv2_layer limit=1
    #pragma HLS ALLOCATION function instances=conv3_layer limit=1
    #pragma HLS ALLOCATION function instances=maxpool_3ch limit=1
    #pragma HLS ALLOCATION function instances=maxpool_9ch_14 limit=1
    #pragma HLS ALLOCATION function instances=maxpool_9ch_7 limit=1
    #pragma HLS ALLOCATION function instances=fc1_layer limit=1
    #pragma HLS ALLOCATION function instances=fc2_layer limit=1
    #pragma HLS ALLOCATION function instances=fc3_layer limit=1
    
    // Reshape input
    input_reshape(input, INPUT);
    
    // Layer 1: Conv1 + ReLU + MaxPool
    padding_input(INPUT, PADDED_INPUT);
    wreshape_conv1(conv1_weight, WBUF_CONV1);
    conv1_layer(PADDED_INPUT, WBUF_CONV1, CONV1_OUT, conv1_bias);
    relu_3ch(CONV1_OUT, CONV1_OUT);
    maxpool_3ch(CONV1_OUT, POOL1_OUT);
    
    // Layer 2: Conv2 + ReLU + MaxPool
    padding_3ch(POOL1_OUT, PADDED_CONV1);
    wreshape_conv2(conv2_weight, WBUF_CONV2);
    conv2_layer(PADDED_CONV1, WBUF_CONV2, CONV2_OUT, conv2_bias);
    relu_9ch_14(CONV2_OUT, CONV2_OUT);
    maxpool_9ch_14(CONV2_OUT, POOL2_OUT);
    
    // Layer 3: Conv3 + ReLU + MaxPool
    padding_9ch(POOL2_OUT, PADDED_CONV2);
    wreshape_conv3(conv3_weight, WBUF_CONV3);
    conv3_layer(PADDED_CONV2, WBUF_CONV3, CONV3_OUT, conv3_bias);
    relu_9ch_7(CONV3_OUT, CONV3_OUT);
    maxpool_9ch_7(CONV3_OUT, POOL3_OUT);
    
    // Flatten
    flatten(POOL3_OUT, FLATTENED);
    
    // Fully connected layers
    fc1_layer(FLATTENED, fc1_weight, fc1_bias, FC1_OUT);
    fc2_layer(FC1_OUT, fc2_weight, fc2_bias, FC2_OUT);
    fc3_layer(FC2_OUT, fc3_weight, fc3_bias, FC3_OUT);
    
    // Output
    output_reshape(FC3_OUT, output);
}