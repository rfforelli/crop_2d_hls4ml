//Numpy array shape [1]
//Min 0.125000000000
//Max 0.125000000000
//Number of zeros 0

#ifndef S16_H_
#define S16_H_

#ifndef __SYNTHESIS__
exponent_scale16_t s16[16];
#else
exponent_scale16_t s16[16] = {{1.0, -3}, {1.0, -3}, {1.0, -3}, {1.0, -3}, {1.0, -3}, {1.0, -3}, {1.0, -3}, {1.0, -3}, {1.0, -3}, {1.0, -3}, {1.0, -3}, {1.0, -3}, {1.0, -3}, {1.0, -3}, {1.0, -3}, {1.0, -3}};
#endif

#endif
