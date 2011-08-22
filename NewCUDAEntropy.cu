

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#if defined(__unix) || defined(__linux)
#include <sys/time.h>
#endif

// includes, project
//#include <cuda.h>
//#include <cuda_runtime.h>
//#include <cufft.h>

#include "cutil_inline.h"

extern "C" void NewCUDAEntropy(float* input, float beta, float tau, int dim[3], int nc, int time);

static __inline__ __device__ float atomicAdd(float *addr, float val)
{
  float old=*addr;
  float assumed;
  do{
    assumed=old;
    old=__int_as_float(atomicCAS((int*)addr, __float_as_int(assumed), __float_as_int(val+assumed)));
  }while(assumed!=old);
  return old;
}

__global__ void computeEntropy(float* input, float* output, int dim0, int dim1, int dim2, float beta, float tau){
  unsigned int voxel=blockIdx.y*gridDim.x+blockIdx.x;
  if (voxel*blockDim.x+threadIdx.x>=dim0*dim1*dim2*blockDim.x)
    return;
  __shared__ float sum;
  float val=expf(-beta*input[voxel*blockDim.x+threadIdx.x]);//nc==blockDim.x
  if (threadIdx.x==0)
    sum=0.0;
  __syncthreads();
  atomicAdd(&sum, val);
  __syncthreads();
  if (sum>1e-6){
    __shared__ float entropy;
    if (threadIdx.x==0)
      entropy=0.0;
    __syncthreads();
    float p=val/sum;
    if (p>1e-6)
      atomicAdd(&entropy, p*logf(p));
    __syncthreads();
    if (threadIdx.x==0){
      output[voxel]=expf(tau*entropy);
    }
  }
  else {
    if (threadIdx.x==0)
      output[voxel]=1.0;
  }
}

void NewCUDAEntropy(float* input, float beta, float tau, int dim[3], int nc, int time){

#if defined(__unix) || defined(__linux)

  struct timeval start, end, s, e;
  double starttime, endtime, stime, etime;

  if (time){
    gettimeofday(&s, NULL);
    stime=s.tv_sec+s.tv_usec/1000000.0;
    gettimeofday(&start, NULL);
    starttime=start.tv_sec+start.tv_usec/1000000.0;
  }

#endif

  float* d_in;
  cudaMalloc((void**) &d_in, dim[0]*dim[1]*dim[2]*nc*sizeof(float));
  float* d_out;//only using this because cudaMemcpy took 1.5 seconds after the kernel, and that's just ridiculous
  cudaMalloc((void**) &d_out, dim[0]*dim[1]*dim[2]*sizeof(float));

  cudaMemcpy(d_in, input, dim[0]*dim[1]*dim[2]*nc*sizeof(float), cudaMemcpyHostToDevice);

  unsigned int nt=((unsigned int)dim[0])*dim[1]*dim[2];
  //lowest power of 2 above nt
  nt--;
  nt|=(nt>>1);
  nt|=(nt>>2);
  nt|=(nt>>4);
  nt|=(nt>>8);
  nt|=(nt>>16);
  nt++;
  if (nt>=1){
    nt--;
    //number of ones in nt--this is equal to log2(nt) because nt was a power of 2 before (this is as tricky as it gets!)
    nt -= ((nt >> 1) & 0x55555555);
    nt = (((nt >> 2) & 0x33333333) + (nt & 0x33333333));
    nt = (((nt >> 4) + nt) & 0x0f0f0f0f);
    nt += (nt >> 8);
    nt += (nt >> 16);
    nt=(nt & 0x0000003f);
  }
  //we make the reasonable assumption here that (dim[0]*dim[1]*dim[2])<=2^32...this is only called "reasonable" for a reason!
  //recall the division algorithm: let L=log2(ceilpow2(dim[0]*dim[1]*dim[2])). L in Z implies L=16q+r for some q in Z, 0<=r<16. Hence, r=L(mod 16), q=(L-r)/16.
  dim3 grid(1<<(nt/2), 1<<(nt-(nt/2)));
  dim3 blocks(nc, 1, 1);

#if defined(__unix) || defined(__linux)

    if (time){
      gettimeofday(&end, NULL);
      endtime=end.tv_sec+(end.tv_usec/1000000.0);
      fprintf(stderr, "\n   cudaMalloc/cudaMemcpy/nt calculations took %.6f seconds.\n", endtime-starttime);
      gettimeofday(&start, NULL);
      starttime=start.tv_sec+start.tv_usec/1000000.0;
    }

#endif

    computeEntropy <<< grid, blocks >>> (d_in, d_out, dim[0], dim[1], dim[2], beta, tau);

#if defined(__unix) || defined(__linux)

    if (time){
      gettimeofday(&end, NULL);
      endtime=end.tv_sec+(end.tv_usec/1000000.0);
      fprintf(stderr, "\n   Kernel took %.6f seconds.\n", endtime-starttime);
      gettimeofday(&start, NULL);
      starttime=start.tv_sec+start.tv_usec/1000000.0;
    }

#endif

  cudaMemcpy(input, d_out, dim[0]*dim[1]*dim[2]*sizeof(float), cudaMemcpyDeviceToHost);

#if defined(__unix) || defined(__linux)

    if (time){
      gettimeofday(&end, NULL);
      endtime=end.tv_sec+(end.tv_usec/1000000.0);
      fprintf(stderr, "\n   cudaMemcpy took %.6f seconds.\n", endtime-starttime);
      gettimeofday(&start, NULL);
      starttime=start.tv_sec+start.tv_usec/1000000.0;
    }

#endif

  cudaFree(d_in);
  cudaFree(d_out);

#if defined(__unix) || defined(__linux)

  if (time){
    gettimeofday(&e, NULL);
    etime=e.tv_sec+e.tv_usec/1000000.0;
    fprintf(stderr, "\n  Full thing took %.6f seconds.\n", etime-stime); 
  }

#endif

}
