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

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <jpeglib.h>

#include <utility>  //for OpenCL
#include <iterator>
#include <string>

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/stitching.hpp"
#include <sstream>
#include <iomanip>


#include <unistd.h>  // NOLINT(build/include_order)
#include <CL/cl.hpp>

#include "tensorflow/contrib/lite/examples/sr/bitmap_helpers.h"

#define LOG(x) std::cerr

#define MEM_SIZE (128)
#define MAX_SOURCE_SIZE (0x100000)

#define widthA 128
#define heightA 128
 
#define widthB heightA
#define heightB 128
 
#define widthC widthA
#define heightC heightB





namespace tflite {
namespace sr {

std::vector<uint8_t> decode_bmp(const uint8_t* input, int row_size, int width,
                                int height, int channels, bool top_down) {
  std::vector<uint8_t> output(height * width * channels);
  for (int i = 0; i < height; i++) {
    int src_pos;
    int dst_pos;

    for (int j = 0; j < width; j++) {
      if (!top_down) {
        src_pos = ((height - 1 - i) * row_size) + j * channels;
      } else {
        src_pos = i * row_size + j * channels;
      }

      dst_pos = (i * width + j) * channels;

      switch (channels) {
        case 1:
          output[dst_pos] = input[src_pos];
          break;
        case 3:
          // BGR -> RGB
          output[dst_pos] = input[src_pos + 2];
          output[dst_pos + 1] = input[src_pos + 1];
          output[dst_pos + 2] = input[src_pos];
          break;
        case 4:
          // BGRA -> RGBA
          output[dst_pos] = input[src_pos + 2];
          output[dst_pos + 1] = input[src_pos + 1];
          output[dst_pos + 2] = input[src_pos];
          output[dst_pos + 3] = input[src_pos + 3];
          break;
        default:
          LOG(FATAL) << "Unexpected number of channels: " << channels;
          break;
      }
    }
  }
  return output;
}

std::vector<uint8_t> read_bmp(const std::string& input_bmp_name, int* width,
                              int* height, int* channels, Settings* s) {
  int begin, end;

  std::ifstream file(input_bmp_name, std::ios::in | std::ios::binary);
  if (!file) {
    LOG(FATAL) << "input file " << input_bmp_name << " not found\n";
    exit(-1);
  }

  begin = file.tellg();
  file.seekg(0, std::ios::end);
  end = file.tellg();
  size_t len = end - begin;

  if (s->verbose) LOG(INFO) << "len: " << len << "\n";

  std::vector<uint8_t> img_bytes(len);
  file.seekg(0, std::ios::beg);
  file.read(reinterpret_cast<char*>(img_bytes.data()), len);
  const int32_t header_size =
      *(reinterpret_cast<const int32_t*>(img_bytes.data() + 10));
  *width = *(reinterpret_cast<const int32_t*>(img_bytes.data() + 18));
  *height = *(reinterpret_cast<const int32_t*>(img_bytes.data() + 22));
  const int32_t bpp =
      *(reinterpret_cast<const int32_t*>(img_bytes.data() + 28));
  *channels = bpp / 8;

  if (s->verbose)
    LOG(INFO) << "width, height, channels: " << *width << ", " << *height
              << ", " << *channels << "\n";

  // there may be padding bytes when the width is not a multiple of 4 bytes
  // 8 * channels == bits per pixel
  const int row_size = (8 * *channels * *width + 31) / 32 * 4;

  // if height is negative, data layout is top down
  // otherwise, it's bottom up
  bool top_down = (*height < 0);

  // Decode image, allocating tensor once the image size is known
  const uint8_t* bmp_pixels = &img_bytes[header_size];
  return decode_bmp(bmp_pixels, row_size, *width, abs(*height), *channels,
                    top_down);
}

std::vector<float> rgb2ycbcr(std::vector<uint8_t> in){
  std::vector<float> result;
  std::vector<uint8_t> temp = in;
  float f_red, f_blue, f_green, pushedin_v;

  for (std::vector<uint8_t>::const_iterator i = temp.begin(); i < temp.end(); i=i+3){
    f_red = (float)*i / 255;
    f_green = (float)*(i+1) / 255;
    f_blue = (float)*(i+2) / 255;

    pushedin_v = (float)(0.2989 * f_red + 0.5866 * f_green + 0.1145 * f_blue);   // Get Y channel value
    result.push_back(pushedin_v);
    //std::cout << pushedin_v << ' ';
    pushedin_v = (float)(-0.1687 * f_red - 0.3313 * f_green + 0.5000 * f_blue);  // Get cb channel value
    result.push_back(pushedin_v);
    pushedin_v = (float)(0.5000 * f_red - 0.4184 * f_green - 0.0816 * f_blue);   // Get cr channel value
    result.push_back(pushedin_v);

    }

  return result;
}


static float Min(float a, float b) {
	return a <= b ? a : b;
}

static float Max(float a, float b) {
	return a >= b ? a : b;
}

std::vector<uint8_t> ycbcr2rgb(std::vector<float> in){
  std::vector<uint8_t> result;
  float f_red, f_blue, f_green, pushedin_v;

  for (int i = 0; i <= in.size(); i=i+3){


    f_red = Max(0.0f, Min(1.0f, (float)(in[i] + 0.0000 * in[i+1] + 1.4022 * in[i+2])));   // Get R channel value in float format
    result.push_back((uint8_t)(f_red * 255));
    //std::cout << pushedin_v << ' ';
    f_green = Max(0.0f, Min(1.0f, (float)(in[i] - 0.3456 * in[i+1] - 0.7145 * in[i+2])));   // Get G channel value in float format
    result.push_back((uint8_t)(f_green * 255));
    f_blue = Max(0.0f, Min(1.0f, (float)(in[i] + 1.7710 * in[i+1] + 0.0000 * in[i+2])));   // Get B channel value in float format
    result.push_back((uint8_t)(f_blue * 255));

    }

  return result;
}


std::vector<float> get_y_channel(std::vector<float>* in){
  std::vector<float> result;
  std::vector<float> temp = *in;

  for (std::vector<float>::const_iterator i = temp.begin(); i < temp.end(); i=i+3){
    result.push_back(*i);
    }

  return result;
}

bool save_jpg_to_file_old(std::vector<uint8_t>* in, int* width, int* height, int* channels) {
  bool try_use_gpu = false;
  bool divide_images = false;
  cv::Stitcher::Mode mode = cv::Stitcher::PANORAMA;
  string result_name = "img_stitched.jpg";
  struct jpeg_compress_struct cinfo;

  struct jpeg_error_mgr jerr;

  std::vector<uint8_t> temp = *in;

  JSAMPROW row_pointer[1];

  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_compress(&cinfo);

  FILE* outfile = fopen("img_output.jpg", "wb");
 
  if (!outfile){
    std::cout << "%s:%d Failed to open output file" << std::endl;
    return false;
  }
  
  

  jpeg_stdio_dest(&cinfo, outfile);


  cinfo.image_width = *width;
  cinfo.image_height = *height;
  cinfo.input_components = *channels;
  cinfo.in_color_space = JCS_RGB;

  int img_width = *width;

  jpeg_set_defaults(&cinfo);
  jpeg_set_quality(&cinfo, 100, TRUE);

  jpeg_start_compress(&cinfo, TRUE);

  unsigned char bytes[img_width * 3];

  int row_number = 0;

  while (cinfo.next_scanline < cinfo.image_height) {
    for (int i = 0;  i < (img_width*3); i+=3){
        bytes[i] = (unsigned char)temp[(img_width*3*row_number + i)];
        bytes[i+1] = (unsigned char)temp[(img_width*3*row_number + i +1)];
        bytes[i+2] = (unsigned char)temp[(img_width*3*row_number + i + 2)];
    }
    row_pointer[0] = bytes;
    (void) jpeg_write_scanlines(&cinfo, row_pointer, 1);
    row_number++;
  }

  jpeg_finish_compress(&cinfo);
  fclose(outfile);
  jpeg_destroy_compress(&cinfo);

  std::cout << "Test image was generated successfully!" << std::endl;

  return true;
}



bool save_jpg_to_file(std::vector<std::vector<uint8_t>>* in, int* ten_width, int* ten_height, int img_n, int img_rn, int img_cn, int strid_w, int strid_h) {
  bool try_use_gpu = false;
  bool divide_images = false;
  int t_h = *ten_height;
  int t_w = *ten_width;
  std::cout << t_h << " " << t_w <<std::endl;
  cv::Stitcher::Mode mode = cv::Stitcher::PANORAMA;
  std::vector<std::vector<uint8_t>> temp = *in;
  string result_name = "result_s.jpg";
  std::vector<cv::Mat> images_r;
  cv::Mat tmp[img_n];
  cv::Mat result;

  int result_w = strid_w*(img_cn - 1) + t_w;
  int result_h = strid_h*(img_rn - 1) + t_h;
  result= cv::Mat(result_h,result_w, CV_8UC3, cv::Scalar(0,0,0));


  int row_n, col_n;
  for(int m_n =0; m_n< temp.size();m_n++){                           //average stitching 
    row_n = m_n/img_cn;                       //get the x y value for temp[m_n]
    col_n = m_n % img_cn;
    
    for (int i=0;i<t_h;i++){                             //grab rgb value from imgs[i_n]
      for(int j=0;j<t_w;j++) {
        if(result.at<cv::Vec3b>((row_n*strid_h+i),(col_n*strid_w+j))[2]==0){
          result.at<cv::Vec3b>((row_n*strid_h+i),(col_n*strid_w+j))[2] = result.at<cv::Vec3b>((row_n*strid_h+i),(col_n*strid_w+j))[2] + temp[m_n][i*t_w*3+j*3];     // put clear r back to imgs[i_n]
          result.at<cv::Vec3b>((row_n*strid_h+i),(col_n*strid_w+j))[1] = result.at<cv::Vec3b>((row_n*strid_h+i),(col_n*strid_w+j))[1] + temp[m_n][i*t_w*3+j*3+1];     // put g back
          result.at<cv::Vec3b>((row_n*strid_h+i),(col_n*strid_w+j))[0] = result.at<cv::Vec3b>((row_n*strid_h+i),(col_n*strid_w+j))[0] + temp[m_n][i*t_w*3+j*3+2];  // put b back
          }
        else{
          result.at<cv::Vec3b>((row_n*strid_h+i),(col_n*strid_w+j))[2] = (result.at<cv::Vec3b>((row_n*strid_h+i),(col_n*strid_w+j))[2] + temp[m_n][i*t_w*3+j*3])/2;
          result.at<cv::Vec3b>((row_n*strid_h+i),(col_n*strid_w+j))[1] = (result.at<cv::Vec3b>((row_n*strid_h+i),(col_n*strid_w+j))[1] + temp[m_n][i*t_w*3+j*3+1])/2;
          result.at<cv::Vec3b>((row_n*strid_h+i),(col_n*strid_w+j))[0] = (result.at<cv::Vec3b>((row_n*strid_h+i),(col_n*strid_w+j))[0] + temp[m_n][i*t_w*3+j*3+2])/2;
        }    
      }
    }
  
  }

  imwrite(result_name, result);
  
  //imwrite("loadtemp.jpg",tmp);
  //for(int m_n =0; m_n< temp.size();m_n++){
  /*for(int m_n =0; m_n< temp.size();m_n++){
    std::cout<< "Temp value for replacing imgae value in picture "<< m_n << std::endl;
    for ( int i = 0; i < 30; i++ ) {                                            
      std::cout << temp[m_n][i] << " " ;
    }
    std::cout<< "======================= "<< m_n << std::endl;
    tmp[m_n] = cv::Mat(t_h,t_w, CV_8UC3, cv::Scalar(0,0,255));
    std::cout<< t_h << " " << t_w << " " << tmp[m_n].cols << " " << tmp[m_n].rows << std::endl;
    //imwrite("enhanced_before.jpg",tmp);
    for (int i=0;i<t_h;i++){                             //grab rgb value from imgs[i_n]
      for(int j=0;j<t_w;j++) {
        //std::cout << (uchar)tmp.at<cv::Vec3b>(i,j)[2] << " " << (uchar)temp[m_n][i*t_w+j]<<" "<<(uchar)tmp.at<cv::Vec3b>(i,j)[1] << " " << (uchar)temp[m_n][i*t_w+j+1]<<" "<<(uchar)tmp.at<cv::Vec3b>(i,j)[0] << " " << (uchar)temp[m_n][i*t_w+j+2]<<std::endl;
        tmp[m_n].at<cv::Vec3b>(i,j)[2] = (uchar)temp[m_n][i*t_w*3+j*3];     // put clear r back to imgs[i_n]
        tmp[m_n].at<cv::Vec3b>(i,j)[1] = (uchar)temp[m_n][i*t_w*3+j*3+1];     // put g back
        tmp[m_n].at<cv::Vec3b>(i,j)[0] = (uchar)temp[m_n][i*t_w*3+j*3+2];     // put b back
      }
    }
  //imwrite("enhanced_last.jpg",tmp);
    std::stringstream ss;
    ss << std::setw(3) << std::setfill('0') << m_n;
    std::string s = ss.str();
    imwrite(("/home/jin/smallpicsforsr/enhanced"+s+".jpg").c_str(),tmp[m_n]);
    images_r.push_back(tmp[m_n]);
  }
  
  for (int h=0;h<images_r.size();h++){
    std::cout<< "Output image " << h << "rbg value: " << std::endl;
    for (int i=0;i<1;i++){                             //grab rgb value from imgs[i_n]
      for(int j=0;j<41;j++) {
        std::cout<< (int)images_r[h].at<cv::Vec3b>(i,j)[2] << " "<<(int)images_r[h].at<cv::Vec3b>(i,j)[1]<<" "<<(int)images_r[h].at<cv::Vec3b>(i,j)[0] <<" ";
      }
    }
    std::cout<<std::endl;
  }*/


  /*cv::Mat pano;
  cv::Stitcher stitcher = cv::Stitcher::createDefault(try_use_gpu);
  stitcher.setRegistrationResol(-1); /// 0.6
  stitcher.setSeamEstimationResol(-1);   /// 0.1
  stitcher.setCompositingResol(-1);   //1
  stitcher.setPanoConfidenceThresh(-1);   //1
  stitcher.setWaveCorrection(true);
  stitcher.setWaveCorrectKind(cv::detail::WAVE_CORRECT_HORIZ);
  cv::Stitcher::Status status = stitcher.stitch(images_r, pano);*/

  /*cv::Mat pano;
  cv::Ptr<cv::Stitcher> stitcher = cv::Stitcher::create(mode, try_use_gpu);
  cv::Stitcher::Status status = stitcher->stitch(images_r, pano);
  if (status != cv::Stitcher::OK)
    {
        std::cout << "Can't stitch images, error code = " << int(status) << std::endl;
    }
  imwrite(result_name, pano);
  std::cout << "stitching completed successfully\n" << result_name << " saved!" << std::endl;

  std::cout << "Test image was generated successfully!" << std::endl;*/

  return true;
}


void test_cv(){
  cv::Stitcher::Mode mode = cv::Stitcher::PANORAMA;
  string result_name = "result_s.jpg";
  std::vector<cv::Mat> images_r;

  for (int i=1;i<=4;i++){
    std::string s = std::to_string(i)+".jpg";
    cv::Mat tmp = cv::imread(s,CV_LOAD_IMAGE_COLOR);
    images_r.push_back(tmp);
  }

}

std::vector<float> replace_y_channel(std::vector<float>* blur_img, float* en_y, int pixel_n){
  std::vector<float> result = *blur_img;
  for(int i=0;i<pixel_n*3;i=i+3){
    result[i]=(*(en_y+(i/3)));
  }
  return result;
}

std::vector<std::vector<uint8_t>>  load_image(Settings* s,int* width, int* height, int ten_width, int ten_height, int strid_w,int strid_h, int* img_n){
  std::vector<std::vector<uint8_t>> in;
  unsigned char *data;
  const char * file_address = s->input_bmp_name.c_str();
  std::vector<uint8_t> result;
  cv::Mat image;
  image = cv::imread(file_address,CV_LOAD_IMAGE_COLOR);
  if(! image.data )  // Check for invalid input
       {
              std::cout <<  "Could not open or find the image" << std::endl ;
       }
  *width = image.cols;
  *height = image.rows;

  std::cout<< *width << " " << *height << std::endl;

  cv::Size smallSize(ten_width,ten_height);
  std::vector<cv::Mat> smallImages;

  for (int x = 0; x < image.rows; x = x + strid_h)
    {
    for (int y = 0; y < image.cols; y = y + strid_w)
      {
        //std::cout <<"Start "<< x << " " << y << std::endl;
        if((y+smallSize.width)>image.cols){
          if((x+smallSize.height)>image.rows){
            cv::Rect rect =  cv::Rect((image.cols-smallSize.width),(image.rows-smallSize.height), smallSize.width, smallSize.height);
            //smallImages.push_back(cv::Mat(image, rect).clone());
          }
          else{
            cv::Rect rect =  cv::Rect((image.cols-smallSize.width),x, smallSize.width, smallSize.height);
            //smallImages.push_back(cv::Mat(image, rect).clone());
          }
        }
        else if((x+smallSize.height)>image.rows){
          cv::Rect rect =  cv::Rect(y,(image.rows-smallSize.height), smallSize.width, smallSize.height);
            //smallImages.push_back(cv::Mat(image, rect).clone());
        }
        else{
          
          cv::Rect rect =  cv::Rect(y,x, smallSize.width, smallSize.height);
          
          smallImages.push_back(cv::Mat(image, rect).clone());
          //std::cout << "End "<<x << " " << y << std::endl;
        }
      }
    }
  *img_n = smallImages.size();

  for(int i_n=0;i_n<smallImages.size();i_n++){
    result.clear();
    for (int i=0;i<smallImages[i_n].rows;i++){
      for(int j=0;j<smallImages[i_n].cols;j++){
        result.push_back((uint8_t)smallImages[i_n].at<cv::Vec3b>(i,j)[2]);
        result.push_back((uint8_t)smallImages[i_n].at<cv::Vec3b>(i,j)[1]);
        result.push_back((uint8_t)smallImages[i_n].at<cv::Vec3b>(i,j)[0]);
      }
    }

    in.push_back(result);
  }

  
  return in;
}


inline void
checkErr(cl_int err, const char * name)
{
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " << name
                 << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}    //check to see if a OpenCL function call has completed successfully

const std::string hw("Hello World\n");  //Used for OpenCL test

void test_OpenCL(){
    cl_int err;
    std::vector< cl::Platform > platformList;
    cl::Platform::get(&platformList);
    checkErr(platformList.size()!=0 ? CL_SUCCESS : -1, "cl::Platform::get");
    std::cerr << "Platform number is: " << platformList.size() << std::endl;
    
    std::string platformVendor;
    platformList[0].getInfo((cl_platform_info)CL_PLATFORM_VENDOR, &platformVendor);
    std::cerr << "Platform is by: " << platformVendor << "\n";
    cl_context_properties cprops[3] = 
        {CL_CONTEXT_PLATFORM, (cl_context_properties)(platformList[0])(), 0};
 
    cl::Context context_test(
       CL_DEVICE_TYPE_GPU, 
       cprops,
       NULL,
       NULL,
       &err);
    checkErr(err, "Conext::Context()");

    char * outH = new char[hw.length()+1];
    cl::Buffer outCL(context_test, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, hw.length()+1, outH, &err);
    checkErr(err, "Buffer::Buffer()");


    std::vector<cl::Device> devices;
    devices = context_test.getInfo<CL_CONTEXT_DEVICES>();
    checkErr(devices.size() > 0 ? CL_SUCCESS : -1, "devices.size() > 0");

    
    /*
    std::ifstream file("test_kernel.cl");
    checkErr(file.is_open() ? CL_SUCCESS:-1, "test_kernel.cl");

    std::string prog( std::istreambuf_iterator<char>(file), (std::istreambuf_iterator<char>()));

    cl::Program::Sources source( 1, std::make_pair(prog.c_str(), prog.length()+1));

    cl::Program program(context, source);
    err = program.build(devices,"");
    checkErr(file.is_open() ? CL_SUCCESS : -1, "Program::build()");

    cl::Kernel kernel(program, "hello", &err);
    checkErr(err, "Kernel::Kernel()");

    err = kernel.setArg(0, outCL);
    checkErr(err, "Kernel::setArg()");

    cl::CommandQueue queue(context, devices[0], 0, &err);
    checkErr(err, "CommandQueue::CommandQueue()");
 

    cl::Event event;
    err = queue.enqueueNDRangeKernel(
        kernel, 
        cl::NullRange,
        cl::NDRange(hw.length()+1),
        cl::NDRange(1, 1), 
        NULL, 
        &event);
    checkErr(err, "ComamndQueue::enqueueNDRangeKernel()");
    event.wait();    
    err = queue.enqueueReadBuffer(
        outCL,
        CL_TRUE,
        0,
        hw.length()+1,
        outH);
    checkErr(err, "ComamndQueue::enqueueReadBuffer()");
    std::cout << outH;
    float executionTimeInMilliseconds = (end - start) * 1.0e-6f;*/

  float * A = (float *)malloc(sizeof(float)*widthA*heightA);
  float * B = (float *)malloc(sizeof(float)*widthB*heightB);
  float * C = (float *)malloc(sizeof(float)*widthC*heightC);
  float * Res = (float *)malloc(sizeof(float)*widthC*heightC);
  float * D= (float *)malloc(sizeof(float)*widthC*heightC);
 
   FILE * fp1 = fopen("matAdata.txt", "w");
  if (!fp1) {
    fprintf(stderr, "Failed to open matAdata.\n");
    exit(1);
   }
 
  for(int i = 0;i < widthA; i++)              //generate random mat A data and write to file 
  {
        for(int j=0;j < heightA;j++){
            float p=(rand()%100)/7.0;
            *(A+i*heightA+j)=rand()%100 + p;
            fprintf(fp1, "%f ",*(A+i*heightA+j));
        }
        fprintf(fp1, "\n");
   }
   fclose(fp1);
 
   fp1 = fopen("matBdata.txt", "w");
   if (!fp1) {
    fprintf(stderr, "Failed to open matAdata.\n");
    exit(1);
   }
 
    for(int i = 0;i < widthB; i++)            //generate random mat B data and write to file 
    {
        for(int j=0; j<heightB; j++) {
            float p=(rand()%100)/7.0;
            *((B+i*heightB+j))=rand()%100 + p;
            fprintf(fp1, "%f ",*(B+i*heightA+j));
        }
        fprintf(fp1, "\n");
    }
    fclose(fp1);
 
  cl_device_id device_id = NULL;
  cl_context context = NULL;
  cl_command_queue command_queue = NULL;
  cl_mem memobjA = NULL;
  cl_mem memobjB = NULL;
  cl_mem memobjC = NULL;
  cl_mem rowA = NULL;
  cl_mem colC = NULL;
  cl_program program = NULL;
  cl_kernel kernel = NULL;
  cl_platform_id platform_id = NULL;
  cl_event event = NULL;
  cl_uint ret_num_devices;
  cl_uint ret_num_platforms;
  cl_int ret;
 
  //char string[MEM_SIZE];
 
  FILE *fp;
  char fileName[] = "test_kernel.cl";
  char *source_str;
  size_t source_size;
  int row = widthA;
  int col = heightC;
  /* Load the source code containing the kernel*/
  fp = fopen(fileName, "r");
  if (!fp) {
    fprintf(stderr, "Failed to load kernel.\n");
    exit(1);
  }
  source_str = (char*)malloc(MAX_SOURCE_SIZE);
  source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
  fclose( fp );
 
  /* Get Platform and Device Info */
  ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
  std::cout << "Platform id is" << platform_id << std::endl;
  ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices);
  std::cout << "Device id is" << device_id << std::endl;
 
  /* Create OpenCL context */
  context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);
 
  /* Create Command Queue */
  command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
 
  /* Create Memory Buffer */
  memobjA = clCreateBuffer(context, CL_MEM_READ_WRITE, widthA * heightA * sizeof(float), NULL, &ret);
  memobjB = clCreateBuffer(context, CL_MEM_READ_WRITE, widthB * heightB * sizeof(float), NULL, &ret);
  memobjC = clCreateBuffer(context, CL_MEM_READ_WRITE, widthC * heightC * sizeof(float), NULL, &ret);
  rowA = clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(int), NULL, &ret);
  colC = clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(int), NULL, &ret);
 
  // Copy the lists A and B to their respective memory buffers
    ret = clEnqueueWriteBuffer(command_queue,memobjA, CL_TRUE, 0,
           widthA * heightA * sizeof(int), A, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, memobjB, CL_TRUE, 0,
            widthB * heightB * sizeof(int), B, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, rowA, CL_TRUE, 0, sizeof(int), &row, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, colC, CL_TRUE, 0, sizeof(int), &col, 0, NULL, NULL);
 
  /* Create Kernel Program from the source */
  program = clCreateProgramWithSource(context, 1, (const char **)&source_str,
                                      (const size_t *)&source_size, &ret);
 
  /* Build Kernel Program */
  ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
 
  /* Create OpenCL Kernel */
  kernel = clCreateKernel(program, "matrixMultiplication", &ret);
 
  /* Set OpenCL Kernel Arguments */
  ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&memobjA);
  ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&memobjB);
  ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&memobjC);
  //ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&memobjA);
  ret = clSetKernelArg(kernel, 3, sizeof(int), (void *)&row);
  ret = clSetKernelArg(kernel, 4, sizeof(int), (void *)&col);
  /* Execute OpenCL Kernel */
  //ret = clEnqueueTask(command_queue, kernel, 0, NULL,NULL);
  size_t globalThreads[2] = {widthA, heightB};
  size_t localThreads[2] = {16,16};
 
  clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, globalThreads, localThreads, NULL, 0, NULL);


  /* Copy results from the memory buffer */
  ret = clEnqueueReadBuffer(command_queue, memobjC, CL_TRUE, 0,
                            widthA * heightC * sizeof(float),Res, 0, NULL, NULL);

  /*profile by check GPU running time*/
  cl_ulong start, end;

  clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
  clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);

  float executionTimeInMilliseconds = (end - start) * 1.0e-6f; 
  std::cout << "GPU running time is " << executionTimeInMilliseconds << " Milliseconds!" <<  std::endl;
  /*Profiling test is over*/
 
  fp1 = fopen("matGPURes.txt", "w");
  if (!fp1) {
    fprintf(stderr, "Failed to open matAdata.\n");
    exit(1);
  }
 
  printf("\nOutput Matirx computation with GPU accelaration to file matGPURes !\n");
    for(int i = 0;i < widthA; i++)
    {
        for(int j=0;j < heightC; j++)
        {
 
            fprintf(fp1, "%f ",*(Res+i*heightC+j));
 
        }
        fprintf(fp1, "\n");
    }
    fclose(fp1);
 
  ret = clFlush(command_queue);
  ret = clFinish(command_queue);
  ret = clReleaseKernel(kernel);
  ret = clReleaseProgram(program);
  ret = clReleaseMemObject(memobjA);
  ret = clReleaseMemObject(memobjB);
  ret = clReleaseMemObject(memobjC);
  ret = clReleaseCommandQueue(command_queue);
  ret = clReleaseContext(context);
 
  free(source_str);
  system("read -p 'Press Enter to continue...' var");        //wait for user's input enter to continue
 
  float sum=0.0;
 
  for(int i = 0;i < widthA; i++)
    {
        for(int j = 0; j < heightC; j++)
        {
            sum = 0;
            for(int k = 0; k < widthB; k++)
            {
                sum += A[i*col+k] * B[k*row+j];
            }
        D[i*heightC+j] = sum;
        }
 
    }
 
    fp1 = fopen("matNorRes.txt", "w");
  if (!fp1) {
    fprintf(stderr, "Failed to open matNorRes.txt file.\n");
    exit(1);
  }
 
  printf("\nOutput Matirx computation without GPU accelaration to file matNorRes for comparison!\n");
    for(int i = 0;i < widthA; i++)
    {
        for(int j=0;j < heightC; j++)
        {
            fprintf(fp1, "%f ",*(D+i*heightC+j));
 
        }
        fprintf(fp1, "\n");
    }
   system("read -p 'Press Enter to continue...' var");

    return;



}

}  // namespace sr
}  // namespace tflite  
