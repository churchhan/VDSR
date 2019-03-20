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

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>
#include <thread>
#include <future>




#include <utility>  //for OpenCL
#include <iterator>

#include <fcntl.h>      // NOLINT(build/include_order)
#include <getopt.h>     // NOLINT(build/include_order)
#include <sys/time.h>   // NOLINT(build/include_order)
#include <sys/types.h>  // NOLINT(build/include_order)
#include <sys/uio.h>    // NOLINT(build/include_order)
#include <unistd.h>     // NOLINT(build/include_order)
#include <CL/cl.hpp>    // for OpenCL





#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/model.h"
#include "tensorflow/contrib/lite/optional_debug_tools.h"
#include "tensorflow/contrib/lite/string_util.h"

#include "tensorflow/contrib/lite/examples/sr/bitmap_helpers.h"
#include "tensorflow/contrib/lite/examples/sr/get_top_n.h"
#include "tensorflow/contrib/lite/examples/sr/invoker.h"

#define LOG(x) std::cerr
#define T_width 128;
#define T_height 128;
#define strid_width 112;
#define strid_height 119;

static const int n_threads = 8;       //number of threads we are using to handle group of small pictures



namespace tflite {
namespace sr {

double get_main_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }    //time function for profiling


void PrintProfilingInfo(const profiling::ProfileEvent* e, uint32_t op_index,
                        TfLiteRegistration registration) {
  // output something like
  // time (ms) , Node xxx, OpCode xxx, symblic name
  //      5.352, Node   5, OpCode   4, DEPTHWISE_CONV_2D

  LOG(INFO) << std::fixed << std::setw(10) << std::setprecision(3)
            << (e->end_timestamp_us - e->begin_timestamp_us) / 1000.0
            << ", Node " << std::setw(3) << std::setprecision(3) << op_index
            << ", OpCode " << std::setw(3) << std::setprecision(3)
            << registration.builtin_code << ", "
            << EnumNameBuiltinOperator(
                   static_cast<BuiltinOperator>(registration.builtin_code))
            << "\n";
}

size_t reverse(size_t n, unsigned int bytes)
    {
        __asm__("BSWAP %0" : "=r"(n) : "0"(n));
        n >>= ((sizeof(size_t) - bytes) * 8);
        n = ((n & 0xaaaaaaaaaaaaaaaa) >> 1) | ((n & 0x5555555555555555) << 1);
        n = ((n & 0xcccccccccccccccc) >> 2) | ((n & 0x3333333333333333) << 2);
        n = ((n & 0xf0f0f0f0f0f0f0f0) >> 4) | ((n & 0x0f0f0f0f0f0f0f0f) << 4);
        return n;
    }

float* m_invoke(Settings* s){
  float* result;
  return result;
}

void RunInference(Settings* s) {
  
  
  if (!s->model_name.c_str()) {
    LOG(ERROR) << "no model file name\n";
    exit(-1);
  }


  int image_width = 0;
  int image_height = 0;
  int image_channels = 3;
  int ten_width = T_width;     //input tensor size
  int ten_height = T_height;
  int strid_w = strid_width;   //stride step on x
  int strid_h = strid_height;  //stride step on y
  int img_n;   //total amount of small pictures need to be handled
  int pix_n = ten_height * ten_width;       //pixel number in a small picture
  int img_rn, img_cn; //small picture row number and col number after split
  

  std::vector<std::vector<uint8_t>> in;
  in = load_image(s, &image_width, &image_height, ten_width, ten_height, strid_w, strid_h, &img_n);  //load image and convert it into vector<uint32_t> rgb group


  /*for (int i =0;i<in.size();i++){
    std::cout<< "Print out in from image============="<<std::endl;
    for ( int j = 0; j < 30; j++ ) {                                            
      std::cout << in[i][j] << " " ;
    }
    std::cout<<std::endl;
  }*/

  std::cout << "image_width :" << image_width << " and image_height : " << image_height << "Img_n is " << img_n << std::endl;

  img_cn = (image_width-ten_width)/strid_w + 1;
  img_rn = (image_height-ten_height)/strid_h + 1;

  std::cout << "img_n = "<< img_n << " img_cn = "<< img_cn<< " img_rn = "<<img_rn << std::endl;
  int total_p = img_cn * img_rn;


  

  std::unique_ptr<tflite::FlatBufferModel> model;
  std::unique_ptr<tflite::FlatBufferModel>* model_address = &model;
  model = tflite::FlatBufferModel::BuildFromFile(s->model_name.c_str());
  if (!model) {
    LOG(FATAL) << "\nFailed to mmap model " << s->model_name << "\n";
    exit(-1);
  }
  LOG(INFO) << "Loaded model " << s->model_name << "\n";
  model->error_reporter();
  LOG(INFO) << "resolved reporter\n";

  

  //========================================================================
  
  std::vector<float*> output_v; 
  


  std::vector< std::vector<uint8_t> > input_g[n_threads];
  std::vector< std::vector<float> >  output[n_threads];

  //divided in into n_threads and rest(could be 0) groups
  int n_one = in.size()/n_threads;   //number of in[] in one group
  for(int i=0;i<n_threads;i++){
    for(int j=0;j<n_one;j++){
      std::cout<<i*n_one+j << " " ;
      input_g[i].push_back(in[i*n_one+j]);
    }
  }
  
  int remainder = in.size()%n_threads;
  std::cout<< "The remainder is "<< in.size()%n_threads << std::endl;
  if( remainder!=0 ){
    for(int i= (in.size()-remainder);i<in.size();i++){
      std::cout<< "i is " << i <<std::endl;
      input_g[n_threads-1].push_back(in[i]);
    }
  }

  //std::cout<< "input_g[0] size is" << input_g[0].size() << "input_g[-1] size is" <<input_g[n_threads-1].size() << std::endl;


  //divided in into n_threads groups ---- end



  struct timeval start_time, stop_time;
  gettimeofday(&start_time, nullptr);

  
  std::thread t[n_threads];

  for (int t_id = 0;t_id<n_threads;t_id++){
      t[t_id] = std::thread(tflite_invoker, model_address, s, input_g[t_id],  &ten_width, &ten_height, &output[t_id]);    //get output data from lite model
  }
  for (int t_id = 0;t_id<n_threads;t_id++){
      t[t_id].join();    
  }
  
  

  gettimeofday(&stop_time, nullptr);
  std::cout << "Total processing time is : " 
              << (get_main_us(stop_time) - get_main_us(start_time)) / (s->loop_count * 1000)
              << " ms \n";



  std::vector<std::vector<uint8_t>> img_out;
  for(int i=0;i<n_threads;i++){
    //std::cout<< "output"<<i<<" size is " << output[i].size()<<std::endl;
      for(int j=0;j<output[i].size();j++){
        std::vector<uint8_t> img_rgb = ycbcr2rgb(output[i][j]);
        img_out.push_back(img_rgb);
      }     
  }              

  bool save_out = save_jpg_to_file(&img_out,&ten_height,&ten_width,img_n, img_rn,img_cn, strid_w, strid_h);


  /*if (s->profiling) {
    profiler->StopProfiling();
    auto profile_events = profiler->GetProfileEvents();
    for (int i = 0; i < profile_events.size(); i++) {
      auto op_index = profile_events[i]->event_metadata;
      const auto node_and_registration =
          interpreter->node_and_registration(op_index);
      const TfLiteRegistration registration = node_and_registration->second;
      PrintProfilingInfo(profile_events[i], op_index, registration);
    }
  }*/


}

void display_usage() {
  LOG(INFO) << "label_image\n"
            << "--accelerated, -a: [0|1], use Android NNAPI or not\n"
            << "--count, -c: loop interpreter->Invoke() for certain times\n"
            << "--input_mean, -b: input mean\n"
            << "--input_std, -s: input standard deviation\n"
            << "--image, -i: image_name.bmp\n"
            << "--labels, -l: labels for the model\n"
            << "--tflite_model, -m: model_name.tflite\n"
            << "--profiling, -p: [0|1], profiling or not\n"
            << "--num_results, -r: number of results to show\n"
            << "--threads, -t: number of threads\n"
            << "--verbose, -v: [0|1] print more information\n"
            << "\n";
}

int Main(int argc, char** argv) {
  Settings s;

  int c;
  while (1) {
    static struct option long_options[] = {
        {"accelerated", required_argument, nullptr, 'a'},
        {"count", required_argument, nullptr, 'c'},
        {"verbose", required_argument, nullptr, 'v'},
        {"image", required_argument, nullptr, 'i'},
        {"labels", required_argument, nullptr, 'l'},
        {"tflite_model", required_argument, nullptr, 'm'},
        {"profiling", required_argument, nullptr, 'p'},
        {"threads", required_argument, nullptr, 't'},
        {"input_mean", required_argument, nullptr, 'b'},
        {"input_std", required_argument, nullptr, 's'},
        {"num_results", required_argument, nullptr, 'r'},
        {nullptr, 0, nullptr, 0}};

    /* getopt_long stores the option index here. */
    int option_index = 0;

    c = getopt_long(argc, argv, "a:b:c:f:i:l:m:p:r:s:t:v:", long_options,
                    &option_index);

    /* Detect the end of the options. */
    if (c == -1) break;

    switch (c) {
      case 'a':
        s.accel = strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'b':
        s.input_mean = strtod(optarg, nullptr);
        break;
      case 'c':
        s.loop_count =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'i':
        s.input_bmp_name = optarg;
        break;
      case 'l':
        s.labels_file_name = optarg;
        break;
      case 'm':
        s.model_name = optarg;
        break;
      case 'p':
        s.profiling =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'r':
        s.number_of_results =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 's':
        s.input_std = strtod(optarg, nullptr);
        break;
      case 't':
        s.number_of_threads = strtol(  // NOLINT(runtime/deprecated_fn)
            optarg, nullptr, 10);
        break;
      case 'v':
        s.verbose =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'h':
      case '?':
        /* getopt_long already printed an error message. */
        display_usage();
        exit(-1);
      default:
        exit(-1);
    }
  }
  //test_OpenCL();
  RunInference(&s);
  return 0;
}

}  // namespace label_image
}  // namespace tflite

int main(int argc, char** argv) {
  
  return tflite::sr::Main(argc, argv);
}
