#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "nnet_utils/nnet_types.h"
#include <cstddef>
#include <cstdio>

// hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 128
#define N_INPUT_2_1 32
#define N_INPUT_3_1 3
#define OUT_HEIGHT_2 1
#define OUT_WIDTH_2 100
#define N_FILT_2 20
#define OUT_HEIGHT_3 98
#define OUT_WIDTH_3 18
#define N_FILT_3 8
#define OUT_HEIGHT_3 98
#define OUT_WIDTH_3 18
#define N_FILT_3 8
#define OUT_HEIGHT_3 98
#define OUT_WIDTH_3 18
#define N_FILT_3 8
#define OUT_HEIGHT_5 49
#define OUT_WIDTH_5 9
#define N_FILT_5 8
#define OUT_HEIGHT_6 47
#define OUT_WIDTH_6 7
#define N_FILT_6 16
#define OUT_HEIGHT_6 47
#define OUT_WIDTH_6 7
#define N_FILT_6 16
#define OUT_HEIGHT_6 47
#define OUT_WIDTH_6 7
#define N_FILT_6 16
#define OUT_HEIGHT_8 23
#define OUT_WIDTH_8 3
#define N_FILT_8 16
#define N_SIZE_0_9 1104
#define N_LAYER_10 16
#define N_LAYER_10 16
#define N_LAYER_10 16
#define N_LAYER_12 10
#define N_LAYER_12 10
#define N_LAYER_12 10

// hls-fpga-machine-learning insert layer-precision
typedef nnet::array<ap_fixed<16,8>, 3*1> input_t;
typedef nnet::array<ap_fixed<16,8>, 3*1> layer2_t;
typedef ap_fixed<16,8> model_default_t;
typedef nnet::array<ap_fixed<16,8>, 8*1> layer3_t;
typedef ap_fixed<4,1> weight3_t;
typedef ap_fixed<4,1> bias3_t;
typedef nnet::array<ap_fixed<16,8>, 8*1> layer14_t;
typedef struct exponent_scale14_t {ap_uint<1> sign;ap_int<2> weight; } exponent_scale14_t;
typedef ap_fixed<4,1> bias14_t;
typedef nnet::array<ap_fixed<16,8>, 8*1> layer4_t;
typedef ap_fixed<18,8> q_conv2d_relu_table_t;
typedef nnet::array<ap_fixed<16,8>, 8*1> layer5_t;
typedef nnet::array<ap_fixed<16,8>, 16*1> layer6_t;
typedef ap_fixed<4,1> weight6_t;
typedef ap_fixed<4,1> bias6_t;
typedef nnet::array<ap_fixed<16,8>, 16*1> layer15_t;
typedef struct exponent_scale15_t {ap_uint<1> sign;ap_int<2> weight; } exponent_scale15_t;
typedef ap_fixed<4,1> bias15_t;
typedef nnet::array<ap_fixed<16,8>, 16*1> layer7_t;
typedef ap_fixed<18,8> q_conv2d_1_relu_table_t;
typedef nnet::array<ap_fixed<16,8>, 16*1> layer8_t;
typedef nnet::array<ap_fixed<16,8>, 16*1> layer10_t;
typedef ap_fixed<4,1> weight10_t;
typedef ap_fixed<4,1> bias10_t;
typedef ap_uint<1> layer10_index;
typedef nnet::array<ap_fixed<16,8>, 16*1> layer16_t;
typedef struct exponent_scale16_t {ap_uint<1> sign;ap_int<4> weight; } exponent_scale16_t;
typedef ap_fixed<4,1> bias16_t;
typedef nnet::array<ap_fixed<16,8>, 16*1> layer11_t;
typedef ap_fixed<18,8> q_dense_relu_table_t;
typedef nnet::array<ap_fixed<16,8>, 10*1> layer12_t;
typedef ap_fixed<4,1> weight12_t;
typedef ap_fixed<4,1> bias12_t;
typedef ap_uint<1> layer12_index;
typedef nnet::array<ap_fixed<16,8>, 10*1> layer17_t;
typedef struct exponent_scale17_t {ap_uint<1> sign;ap_int<2> weight; } exponent_scale17_t;
typedef ap_fixed<4,1> bias17_t;
typedef nnet::array<ap_fixed<16,8>, 10*1> result_t;
typedef ap_fixed<18,8> q_dense_1_softmax_table_t;
typedef ap_fixed<18,8,AP_RND,AP_SAT> q_dense_1_softmax_exp_table_t;
typedef ap_fixed<18,8,AP_RND,AP_SAT> q_dense_1_softmax_inv_table_t;

#endif
