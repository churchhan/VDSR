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

#ifndef TENSORFLOW_CONTRIB_LITE_EXAMPLES_SR_INVOKER_H_
#define TENSORFLOW_CONTRIB_LITE_EXAMPLES_SR_INVOKER_H_

#include "tensorflow/contrib/lite/examples/sr/bitmap_helpers_impl.h"
#include "tensorflow/contrib/lite/examples/sr/sr.h"

namespace tflite {
namespace sr {


void tflite_invoker(std::unique_ptr<tflite::FlatBufferModel>* model_address,Settings* s,std::vector< std::vector<uint8_t> > in, int* t_width, int* t_height,std::vector< std::vector<float> >* out);

}  // namespace sr
}  // namespace tflite

#endif  // TENSORFLOW_CONTRIB_LITE_EXAMPLES_SR_INVOKER_H_
