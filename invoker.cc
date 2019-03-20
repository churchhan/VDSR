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

#include <utility>  //for OpenCL
#include <iterator>

#include <fcntl.h>      // NOLINT(build/include_order)
#include <getopt.h>     // NOLINT(build/include_order)
#include <sys/time.h>   // NOLINT(build/include_order)
#include <sys/types.h>  // NOLINT(build/include_order)
#include <sys/uio.h>    // NOLINT(build/include_order)
#include <unistd.h>     // NOLINT(build/include_order)





#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/model.h"
#include "tensorflow/contrib/lite/optional_debug_tools.h"
#include "tensorflow/contrib/lite/string_util.h"

#include "tensorflow/contrib/lite/examples/sr/bitmap_helpers.h"
#include "tensorflow/contrib/lite/examples/sr/sr.h"

#define LOG(x) std::cerr

namespace tflite {
namespace sr {

double get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

void tflite_invoker(std::unique_ptr<tflite::FlatBufferModel>* model_address,Settings* s,std::vector< std::vector<uint8_t> > input_g, int* t_width, int* t_height,std::vector< std::vector<float> >* out){
    int pix_n = (*t_width) * (*t_height); 
    std::cout<< "pix_n is " << pix_n <<std::endl;
    std::vector< std::vector<float> >::iterator it_output;
    
    std::unique_ptr<tflite::Interpreter> interpreter;  //this function built with unique_ptr, must maintain unique_ptr
    tflite::ops::builtin::BuiltinOpResolver resolver;

    tflite::InterpreterBuilder(**model_address, resolver)(&interpreter); //this function built with unique_ptr, must maintain unique_ptr for model and interpreter important
    if (!interpreter) {
        LOG(FATAL) << "Failed to construct interpreter\n";
        std::cout << "Failed to construct interpreter\n";
        exit(-1);
    }

    interpreter->UseNNAPI(s->accel);

    if (s->number_of_threads != -1) {
        interpreter->SetNumThreads(s->number_of_threads);
    }

    int input = interpreter->inputs()[0];
    if (s->verbose) LOG(INFO) << "input: " << input << "\n";

    const std::vector<int> inputs = interpreter->inputs();
    const std::vector<int> outputs = interpreter->outputs();

    if (s->verbose) {
        LOG(INFO) << "number of inputs: " << inputs.size() << "\n";
        LOG(INFO) << "number of outputs: " << outputs.size() << "\n";
    }

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        LOG(FATAL) << "Failed to allocate tensors!";
    }

    profiling::Profiler* profiler = new profiling::Profiler();
    interpreter->SetProfiler(profiler);

    if (s->profiling) profiler->StartProfiling();

    for(int input_id=0;input_id<input_g.size();input_id++){
        std::vector<float> img_float, img_y;
        img_float = rgb2ycbcr(input_g[input_id]);
        img_y = get_y_channel(&img_float);
        //std::cout<<"Img_float size is "<<img_float.size()<<" Imag_y size is "<< img_y.size()<< std::endl;

        float* input_v = interpreter->typed_input_tensor<float>(0);       //fill data into input tensor
        for ( int i = 0; i < pix_n; i++ ) {                                            //fill data into input tensor
            *(input_v+i) = img_y[i] ;
        }

    //start to invoke to get clear y channel for every small picture
        struct timeval start_time, stop_time;
        gettimeofday(&start_time, nullptr);
        if (interpreter->Invoke() != kTfLiteOk) {
            LOG(FATAL) << "Failed to invoke tflite!\n";
        }
       
        gettimeofday(&stop_time, nullptr);
        std::cout << "Invoker cost time: "
              << (get_us(stop_time) - get_us(start_time)) / (s->loop_count * 1000)
              << " ms \n";

    //end to invoke

    float* output_ptr = interpreter->typed_output_tensor<float>(0);    //get output data from lite model
    std::vector<float> replaced = replace_y_channel(&img_float,output_ptr, pix_n);
    //std::cout<< "replaced size is " << replaced.size()<< std::endl;
    
    out->push_back(replaced);

    }



}


}  // namespace sr
}  // namespace tflite
