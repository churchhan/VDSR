/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_CONTRIB_LITE_EXAMPLES_SR_BITMAP_HELPERS_H_
#define TENSORFLOW_CONTRIB_LITE_EXAMPLES_SR_BITMAP_HELPERS_H_

#include "tensorflow/contrib/lite/examples/sr/bitmap_helpers_impl.h"
#include "tensorflow/contrib/lite/examples/sr/sr.h"

namespace tflite {
namespace sr {

std::vector<uint8_t> read_bmp(const std::string& input_bmp_name, int* width,
                              int* height, int* channels, Settings* s);

template <class T>
void resize(T* out, float* in, int image_height, int image_width,
            int image_channels, int wanted_height, int wanted_width,
            int wanted_channels, Settings* s);

// explicit instantiation
template void resize<float>(float*, float*, int, int, int, int, int,
                            int, Settings*);


std::vector<float> rgb2ycbcr(std::vector<uint8_t> in, int* width, int* height, int* channels);

std::vector<uint8_t> ycbcr2rgb(std::vector<float>* in, int* width, int* height, int* channels); 

std::vector<float> get_y_channel(std::vector<float>* in);

std::vector<float> replace_y_channel(std::vector<float>* blur_img, float* en_y, int pixel_n);

bool save_jpg_to_file(char *file_name, std::vector<uint8_t>* in);

bool save_jpg_to_file_old(std::vector<uint8_t>* in, int* width, int* height, int* channels);

bool save_jpg_to_file(std::vector<std::vector<uint8_t>>* in, int* width, int* height, int img_n ,int img_rn, int img_cn, int strid);

std::vector<std::vector<uint8_t>>  get_jpg_size(Settings* s,int* width, int* height, int ten_width, int ten_height, int strid, int* img_n);

void test_OpenCL();

}  // namespace label_image
}  // namespace tflite

#endif  // TENSORFLOW_CONTRIB_LITE_EXAMPLES_LABEL_IMAGE_BITMAP_HELPERS_H_
