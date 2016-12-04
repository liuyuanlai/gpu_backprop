#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <sys/time.h>
#include <string.h>
// includes, kernels
#include "backprop_cuda_kernel.cu"
#include "backprop.h"




void *__gxx_personality_v0;

////////////////////////////////////////////////////////////////////////////////

extern "C"
void bpnn_layerforward(float *l1, float *l2, float **conn, int n1, int n2);

extern "C"
void bpnn_output_error(float *delta, float *target, float *output, int nj, float *err);

extern "C"
void bpnn_hidden_error(float *delta_h, int nh, float *delta_o, int no, float **who, float *hidden, float *err);

extern "C" 
void bpnn_adjust_weights(float *delta, int ndelta, float *ly, int nly, float **w, float **oldw);



extern "C" 
void startTime(Timer* timer);


extern "C" 
void stopTime(Timer* timer);


extern "C" 
float elapsedTime(Timer timer);


extern "C"
int setup(int argc, char** argv);

extern "C"
float **alloc_2d_dbl(int m, int n);

extern "C"
float squash(float x);

double gettime() {
  struct timeval t;
  gettimeofday(&t,NULL);
  return t.tv_sec+t.tv_usec*1e-6;
}

unsigned int num_threads = 0;
unsigned int num_blocks = 0;

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{
  setup(argc, argv);
}


extern "C"
void bpnn_train_cuda(BPNN *net, float *eo, float *eh)
{
  int in, hid, out;
  float out_err, hid_err;

  Timer timer, timer_forward, timer_adjust, timer_total;
  
  in = net->input_n;
  hid = net->hidden_n;
  out = net->output_n;   
   
  int m = 0;
  float *input_hidden_cuda;
  float *input_cuda;
  float *output_hidden_cuda;
  float *partial_sum;
  float *hidden_partial_sum;
  float *hidden_delta_cuda;
  float *input_prev_weights_cuda;
  float sum;
  float *input_weights_one_dim;
  float *input_weights_prev_one_dim;
  num_blocks = in / BLOCK_HEIGHT;  
  dim3  grid(GRID_WIDTH, GRID_HEIGHT);
  dim3  threads(BLOCK_WIDTH, BLOCK_HEIGHT);
  
  input_weights_one_dim = (float *) malloc((in + 1)* (hid + 1) * sizeof(float));
  input_weights_prev_one_dim = (float *) malloc((in + 1)* (hid + 1) * sizeof(float));
  partial_sum = (float *) malloc(num_blocks * BLOCK_WIDTH * sizeof(float));
 
  // this preprocessing stage is added to correct the bugs of wrong memcopy using two-dimensional net->inputweights
  for (int k = 0; k <= in; k++) { 
   for (int j = 0; j <= hid; j++) {
    input_weights_one_dim[m] = net->input_weights[k][j];
    input_weights_prev_one_dim[m] = net-> input_prev_weights[k][j];
    m++;
    }
  }
  
  cudaMalloc((void**) &input_cuda, (in + 1) * sizeof(float));
  cudaMalloc((void**) &output_hidden_cuda, (hid + 1) * sizeof(float));
  cudaMalloc((void**) &input_hidden_cuda, (in + 1) * (hid + 1) * sizeof(float));
  cudaMalloc((void**) &hidden_partial_sum, num_blocks * BLOCK_WIDTH * sizeof(float));
 
  printf("Performing GPU computation\n");
  printf("Performing first layer forward\n");
  startTime(&timer_total);
  startTime(&timer_forward);
  //printf("in= %d, hid = %d, numblocks = %d\n", in, hid, num_blocks);
  printf("Copying data from host to device..."); fflush(stdout);
  startTime(&timer);
  cudaMemcpy(input_cuda, net->input_units, (in + 1) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(input_hidden_cuda, input_weights_one_dim, (in + 1) * (hid + 1) * sizeof(float), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  stopTime(&timer); printf("%f s\n", elapsedTime(timer));


  printf("Launching kernel for first layer forward..."); fflush(stdout);
  startTime(&timer);
  bpnn_layerforward_CUDA<<< grid, threads >>>(input_cuda,
                                            output_hidden_cuda,
                                            input_hidden_cuda,
                                            hidden_partial_sum,
                                            in,
                                            hid);

  stopTime(&timer); printf("%f s\n", elapsedTime(timer));
  cudaThreadSynchronize();
  
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("bpnn kernel error: %s\n", cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }

  printf("Copying data from device to host..."); fflush(stdout);
  startTime(&timer);
  cudaMemcpy(partial_sum, hidden_partial_sum, num_blocks * BLOCK_WIDTH * sizeof(float), cudaMemcpyDeviceToHost);
  stopTime(&timer); printf("%f s\n", elapsedTime(timer));

  printf("after reduction..."); fflush(stdout);
  startTime(&timer);
  for (int j = 1; j <= hid; j++) {
    sum = 0.0;
    for (int k = 0; k < num_blocks; k++) {  
      sum += partial_sum[k * hid + j-1] ;
    }
  sum += net->input_weights[0][j];
  net-> hidden_units[j] = float(1.0 / (1.0 + exp(-sum)));
  }
  stopTime(&timer); printf("%f s\n", elapsedTime(timer));
  stopTime(&timer_forward); printf("Total time for fist layer forward...%f s\n", elapsedTime(timer_forward));

  bpnn_layerforward(net->hidden_units, net->output_units, net->hidden_weights, hid, out);
  bpnn_output_error(net->output_delta, net->target, net->output_units, out, &out_err);
  bpnn_hidden_error(net->hidden_delta, hid, net->output_delta, out, net->hidden_weights, net->hidden_units, &hid_err);  
  bpnn_adjust_weights(net->output_delta, out, net->hidden_units, hid, net->hidden_weights, net->hidden_prev_weights);

  printf("Performing weight adjust\n");
  startTime(&timer_adjust);

  cudaMalloc((void**) &hidden_delta_cuda, (hid + 1) * sizeof(float));
  cudaMalloc((void**) &input_prev_weights_cuda, (in + 1) * (hid + 1) * sizeof(float));

  printf("Copying data from host to device..."); fflush(stdout);
  startTime(&timer);
  cudaMemcpy(hidden_delta_cuda, net->hidden_delta, (hid + 1) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(input_prev_weights_cuda, input_weights_prev_one_dim, (in + 1) * (hid + 1) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(input_hidden_cuda, input_weights_one_dim, (in + 1) * (hid + 1) * sizeof(float), cudaMemcpyHostToDevice);
  stopTime(&timer); printf("%f s\n", elapsedTime(timer));

  printf("Launching kernel for weight adjust..."); fflush(stdout);
  startTime(&timer);
  bpnn_adjust_weights_cuda<<< grid, threads >>>(hidden_delta_cuda,  
                        hid, 
                        input_cuda, 
                        in,
                        input_hidden_cuda, 
                        input_prev_weights_cuda
                        );
  stopTime(&timer); printf("%f s\n", elapsedTime(timer));

  printf("Copying data from device to host..."); fflush(stdout);
  startTime(&timer);
  //cudaMemcpy(net->input_units, input_cuda, (in + 1) * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(input_weights_one_dim, input_hidden_cuda, (in + 1) * (hid + 1) * sizeof(float), cudaMemcpyDeviceToHost);
  stopTime(&timer); printf("%f s\n", elapsedTime(timer));

  stopTime(&timer_adjust); printf("Total time for weight adjust...%f s\n", elapsedTime(timer_adjust));

  cudaFree(input_cuda);
  cudaFree(output_hidden_cuda);
  cudaFree(input_hidden_cuda);
  cudaFree(hidden_partial_sum);
  cudaFree(input_prev_weights_cuda);
  cudaFree(hidden_delta_cuda);
  
  stopTime(&timer_total); printf("Total time for bp...%f s\n", elapsedTime(timer_total));

  free(partial_sum);
  free(input_weights_one_dim);
  free(input_weights_prev_one_dim);  
  
  
  

}