#include <iostream>

#include "myproject.h"
#include "parameters.h"

void myproject(
    hls::stream<input_t> &input_1,
    hls::stream<result_t> &layer13_out
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS INTERFACE axis port=input_1,layer13_out 
    #pragma HLS DATAFLOW 

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        // hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<weight3_t, 216>(w3, "w3.txt");
        nnet::load_weights_from_txt<bias3_t, 8>(b3, "b3.txt");
        nnet::load_exponent_weights_from_txt<exponent_scale14_t, 8>(s14, "s14.txt");
        nnet::load_weights_from_txt<bias14_t, 8>(b14, "b14.txt");
        nnet::load_weights_from_txt<weight6_t, 1152>(w6, "w6.txt");
        nnet::load_weights_from_txt<bias6_t, 16>(b6, "b6.txt");
        nnet::load_exponent_weights_from_txt<exponent_scale15_t, 16>(s15, "s15.txt");
        nnet::load_weights_from_txt<bias15_t, 16>(b15, "b15.txt");
        nnet::load_weights_from_txt<weight10_t, 17664>(w10, "w10.txt");
        nnet::load_weights_from_txt<bias10_t, 16>(b10, "b10.txt");
        nnet::load_exponent_weights_from_txt<exponent_scale16_t, 16>(s16, "s16.txt");
        nnet::load_weights_from_txt<bias16_t, 16>(b16, "b16.txt");
        nnet::load_weights_from_txt<weight12_t, 160>(w12, "w12.txt");
        nnet::load_weights_from_txt<bias12_t, 10>(b12, "b12.txt");
        nnet::load_exponent_weights_from_txt<exponent_scale17_t, 10>(s17, "s17.txt");
        nnet::load_weights_from_txt<bias17_t, 10>(b17, "b17.txt");
        loaded_weights = true;
    }
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    hls::stream<layer2_t> layer2_out("layer2_out");
    #pragma HLS STREAM variable=layer2_out depth=2000
    nnet::crop_2d<input_t, layer2_t, config2>(input_1, layer2_out); // cropping2d

    hls::stream<layer3_t> layer3_out("layer3_out");
    #pragma HLS STREAM variable=layer3_out depth=1764
    nnet::conv_2d_cl<layer2_t, layer3_t, config3>(layer2_out, layer3_out, w3, b3); // q_conv2d

    hls::stream<layer14_t> layer14_out("layer14_out");
    #pragma HLS STREAM variable=layer14_out depth=1764
    nnet::normalize<layer3_t, layer14_t, config14>(layer3_out, layer14_out, s14, b14); // q_conv2d_alpha

    hls::stream<layer4_t> layer4_out("layer4_out");
    #pragma HLS STREAM variable=layer4_out depth=1764
    nnet::relu<layer14_t, layer4_t, relu_config4>(layer14_out, layer4_out); // q_conv2d_relu

    hls::stream<layer5_t> layer5_out("layer5_out");
    #pragma HLS STREAM variable=layer5_out depth=441
    nnet::pooling2d_cl<layer4_t, layer5_t, config5>(layer4_out, layer5_out); // max_pooling2d

    hls::stream<layer6_t> layer6_out("layer6_out");
    #pragma HLS STREAM variable=layer6_out depth=329
    nnet::conv_2d_cl<layer5_t, layer6_t, config6>(layer5_out, layer6_out, w6, b6); // q_conv2d_1

    hls::stream<layer15_t> layer15_out("layer15_out");
    #pragma HLS STREAM variable=layer15_out depth=329
    nnet::normalize<layer6_t, layer15_t, config15>(layer6_out, layer15_out, s15, b15); // q_conv2d_1_alpha

    hls::stream<layer7_t> layer7_out("layer7_out");
    #pragma HLS STREAM variable=layer7_out depth=329
    nnet::relu<layer15_t, layer7_t, relu_config7>(layer15_out, layer7_out); // q_conv2d_1_relu

    hls::stream<layer8_t> layer8_out("layer8_out");
    #pragma HLS STREAM variable=layer8_out depth=69
    nnet::pooling2d_cl<layer7_t, layer8_t, config8>(layer7_out, layer8_out); // max_pooling2d_1

    auto& layer9_out = layer8_out;
    hls::stream<layer10_t> layer10_out("layer10_out");
    #pragma HLS STREAM variable=layer10_out depth=1
    nnet::dense<layer8_t, layer10_t, config10>(layer9_out, layer10_out, w10, b10); // q_dense

    hls::stream<layer16_t> layer16_out("layer16_out");
    #pragma HLS STREAM variable=layer16_out depth=1
    nnet::normalize<layer10_t, layer16_t, config16>(layer10_out, layer16_out, s16, b16); // q_dense_alpha

    hls::stream<layer11_t> layer11_out("layer11_out");
    #pragma HLS STREAM variable=layer11_out depth=1
    nnet::relu<layer16_t, layer11_t, relu_config11>(layer16_out, layer11_out); // q_dense_relu

    hls::stream<layer12_t> layer12_out("layer12_out");
    #pragma HLS STREAM variable=layer12_out depth=1
    nnet::dense<layer11_t, layer12_t, config12>(layer11_out, layer12_out, w12, b12); // q_dense_1

    hls::stream<layer17_t> layer17_out("layer17_out");
    #pragma HLS STREAM variable=layer17_out depth=1
    nnet::normalize<layer12_t, layer17_t, config17>(layer12_out, layer17_out, s17, b17); // q_dense_1_alpha

    nnet::softmax<layer17_t, result_t, softmax_config13>(layer17_out, layer13_out); // q_dense_1_softmax

}
