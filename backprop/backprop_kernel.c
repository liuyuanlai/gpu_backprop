#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#include "backprop.h"

////////////////////////////////////////////////////////////////////////////////

//extern void bpnn_layerforward(float *l1, float *l2, float **conn, int n1, int n2);

//extern void bpnn_output_error(float *delta, float *target, float *output, int nj, float *err);

//extern void bpnn_hidden_error(float *delta_h, int nh, float *delta_o, int no, float **who, float *hidden, float *err);

//extern void bpnn_adjust_weights(float *delta, int ndelta, float *ly, int nly, float **w, float **oldw);


__gloal__ void bpnn_layerforward(l1, l2, conn, n1, n2)
float *l1, *l2, **conn;
int n1, n2;
{
  float sum;
  int j, k;

  /*** Set up thresholding unit ***/
  l1[0] = 1.0;
  /*** For each unit in second layer ***/
  for (j = 1; j <= n2; j++) {

    /*** Compute weighted sum of its inputs ***/
    sum = 0.0;
    for (k = 0; k <= n1; k++) { 
      sum += conn[k][j] * l1[k]; 
    }
    l2[j] = squash(sum);
  }
}

//extern "C"
__global__ void bpnn_output_error(delta, target, output, nj, err)  
float *delta, *target, *output, *err;
int nj;
{
  int j;
  float o, t, errsum;
  errsum = 0.0;
  for (j = 1; j <= nj; j++) {
    o = output[j];
    t = target[j];
    delta[j] = o * (1.0 - o) * (t - o);
    errsum += ABS(delta[j]);
  }
  *err = errsum;
}


__global__ void bpnn_hidden_error(delta_h,   
             nh, 
             delta_o, 
             no, 
             who, 
             hidden, 
             err)
float *delta_h, *delta_o, *hidden, **who, *err;
int nh, no;
{
  int j, k;
  float h, sum, errsum;

  errsum = 0.0;
  for (j = 1; j <= nh; j++) {
    h = hidden[j];
    sum = 0.0;
    for (k = 1; k <= no; k++) {
      sum += delta_o[k] * who[j][k];
    }
    delta_h[j] = h * (1.0 - h) * sum;
    errsum += ABS(delta_h[j]);
  }
  *err = errsum;
}


__global__ void bpnn_adjust_weights(delta, ndelta, ly, nly, w, oldw)
float *delta, *ly, **w, **oldw;
{
  float new_dw;
  int k, j;
  ly[0] = 1.0;
  //eta = 0.3;
  //momentum = 0.3;

  for (j = 1; j <= ndelta; j++) {
    for (k = 0; k <= nly; k++) {
      new_dw = ((ETA * delta[j] * ly[k]) + (MOMENTUM * oldw[k][j]));
    w[k][j] += new_dw;
    oldw[k][j] = new_dw;
    }
  }
}


extern int setup(int argc, char** argv);

extern float **alloc_2d_dbl(int m, int n);

extern float squash(float x);

double gettime() {
  struct timeval t;
  gettimeofday(&t,NULL);
  return t.tv_sec+t.tv_usec*1e-6;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{
	setup(argc, argv);
}


void bpnn_train_kernel(BPNN *net, float *eo, float *eh)
{
  int in, hid, out;
  float out_err, hid_err;
  
  in = net->input_n;
  hid = net->hidden_n;
  out = net->output_n;   
   
  printf("Performing CPU computation\n");
  bpnn_layerforward(net->input_units, net->hidden_units,net->input_weights, in, hid);
  bpnn_layerforward(net->hidden_units, net->output_units, net->hidden_weights, hid, out);
  bpnn_output_error(net->output_delta, net->target, net->output_units, out, &out_err);
  bpnn_hidden_error(net->hidden_delta, hid, net->output_delta, out, net->hidden_weights, net->hidden_units, &hid_err);  
  bpnn_adjust_weights(net->output_delta, out, net->hidden_units, hid, net->hidden_weights, net->hidden_prev_weights);
  bpnn_adjust_weights(net->hidden_delta, hid, net->input_units, in, net->input_weights, net->input_prev_weights);

}
