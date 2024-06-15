#ifndef NNET_CROP2D_H_
#define NNET_CROP2D_H_

#include "nnet_common.h"
#include "hls_stream.h"

namespace nnet {

struct crop_2d_config {
    static const unsigned n_in = 1;
    static const unsigned n_out = 1;
    static const unsigned in_height = 1;
    static const unsigned in_width = 1;
    static const unsigned out_height = 1;
    static const unsigned out_width = 1;
    static const unsigned n_chan = 1;
    static const unsigned crop_top = 1;
    static const unsigned crop_bottom = 1;
    static const unsigned crop_left = 1;
    static const unsigned crop_right = 1;
};

template<class data_T, class res_T, typename CONFIG_T>
void crop_2d(
    hls::stream<data_T> &data,
    hls::stream<res_T> &res
) {

    ReadInputHeight:
    for (unsigned i_ih = 0; i_ih < CONFIG_T::in_height; i_ih++) {
      ReadInputWidth:
      for (unsigned i_iw = 0; i_iw < CONFIG_T::in_width; i_iw++) {
        #pragma HLS LOOP_FLATTEN
        data_T in_data = data.read();
        if((i_ih >= CONFIG_T::crop_top) && (i_ih < (CONFIG_T::in_height - CONFIG_T::crop_bottom)) && (i_iw >= CONFIG_T::crop_left) && (i_iw < (CONFIG_T::in_width - CONFIG_T::crop_right))){
          res.write(in_data);
        }
      }
    }
  }
}

#endif