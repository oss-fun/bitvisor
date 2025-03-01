#ifndef _KEYSTROKE_H
#define _KEYSTROKE_H

// #include <core/types.h>

#ifdef __cplusplus
extern "C" {
#endif

bool keystroke_init(void);

float keystroke_infer_float(float* input_data, int input_length);

int keystroke_predict(int* features, int feature_count);

#ifdef __cplusplus
}
#endif

#endif /* _KEYSTROKE_H */
