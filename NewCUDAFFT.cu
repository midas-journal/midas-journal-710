
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#if defined(__unix) || defined(__linux)
#include <sys/time.h>
#endif

// includes, project
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>


extern "C" void NewCUDAFFT(float *input, int dim[3], int forward, int doComplex, int time)
{

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

  if (doComplex==1){

    cufftComplex *d_in;
    cufftComplex *d_out;
    cufftComplex *h_in;
    cufftComplex *h_out;

    int x = dim[0];
    int y = dim[1];
    int z = dim[2];

    int size=x*y*z;
    int mem_size=sizeof(cufftComplex)*size;
    h_in=(cufftComplex*)malloc(mem_size);
    h_out=(cufftComplex*)malloc(mem_size);

#if defined(__unix) || defined(__linux)

    if (time){
      gettimeofday(&end, NULL);
      endtime=end.tv_sec+(end.tv_usec/1000000.0);
      fprintf(stderr, "\n   mallocs took %.6f seconds.\n", endtime-starttime);
      gettimeofday(&start, NULL);
      starttime=start.tv_sec+start.tv_usec/1000000.0;
    }

#endif

    for(int i = 0; i < size; i++)//initialize values
      {
    	h_in[i].x = input[2*i];
    	h_in[i].y = input[2*i+1];
      }

#if defined(__unix) || defined(__linux)

    if (time){
      gettimeofday(&end, NULL);
      endtime=end.tv_sec+(end.tv_usec/1000000.0);
      fprintf(stderr, "\n   Populating took %.6f seconds.\n", endtime-starttime);
      gettimeofday(&start, NULL);
      starttime=start.tv_sec+start.tv_usec/1000000.0;
    }

#endif

    cudaMalloc((void**)&d_in, mem_size);//allocate memory on device

    cudaMemcpy(d_in, h_in, mem_size, cudaMemcpyHostToDevice);//copy memory to device

#if defined(__unix) || defined(__linux)

    if (time){
      gettimeofday(&end, NULL);
      endtime=end.tv_sec+(end.tv_usec/1000000.0);
      fprintf(stderr, "\n   cudaMalloc/cudaMemcpy took %.6f seconds.\n", endtime-starttime);
      gettimeofday(&start, NULL);
      starttime=start.tv_sec+start.tv_usec/1000000.0;
    }

#endif

    cufftHandle planForward;//create a plan--much like LAPACK's procedure for optimization

    cufftPlan3d(&planForward, x, y, z, CUFFT_C2C);//initialize plan

    d_out=d_in;

#if defined(__unix) || defined(__linux)

    if (time){
      gettimeofday(&end, NULL);
      endtime=end.tv_sec+(end.tv_usec/1000000.0);
      fprintf(stderr, "\n   Planning took %.6f seconds.\n", endtime-starttime);
      gettimeofday(&start, NULL);
      starttime=start.tv_sec+start.tv_usec/1000000.0;
    }

#endif

    if (forward==1)
      cufftExecC2C(planForward, d_in, d_out, CUFFT_FORWARD);//execute FFT
    else
      cufftExecC2C(planForward, d_in, d_out, CUFFT_INVERSE);//execute RFFT

#if defined(__unix) || defined(__linux)

    if (time){
      gettimeofday(&end, NULL);
      endtime=end.tv_sec+(end.tv_usec/1000000.0);
      fprintf(stderr, "\n   Execution took %.6f seconds.\n", endtime-starttime);
      gettimeofday(&start, NULL);
      starttime=start.tv_sec+start.tv_usec/1000000.0;
    }

#endif

    cudaMemcpy(h_out, d_out, mem_size, cudaMemcpyDeviceToHost);

#if defined(__unix) || defined(__linux)

    if (time){
      gettimeofday(&end, NULL);
      endtime=end.tv_sec+(end.tv_usec/1000000.0);
      fprintf(stderr, "\n   cudaMemcpy took %.6f seconds.\n", endtime-starttime);
      gettimeofday(&start, NULL);
      starttime=start.tv_sec+start.tv_usec/1000000.0;
    }

#endif

    int normalizer;//note: when using CUFFT, upon RFFT a normalization factor of 1/pixels = (pixels^(-1/2))^2 must be applied, since the device doesn't normalize on its own
    if(forward == 0)
      normalizer = size;
    else
      normalizer = 1;

#if defined(__unix) || defined(__linux)

    for(int i=0;i<size;i++)//copy out results
      {
    	input[i*2] = h_out[i].x/normalizer;
    	input[(i*2)+1] = h_out[i].y/normalizer;
      }

    if (time){
      gettimeofday(&end, NULL);
      endtime=end.tv_sec+(end.tv_usec/1000000.0);
      fprintf(stderr, "\n   Copying results took %.6f seconds.\n", endtime-starttime);
      gettimeofday(&start, NULL);
      starttime=start.tv_sec+start.tv_usec/1000000.0;
    }

#endif

    cufftDestroy(planForward);
    cudaFree(d_in);
    free(h_in);
    free(h_out);
  }
  else {

    if (forward==1){

      cufftReal* d_in;
      cufftComplex* d_out;
      cufftReal* h_in;
      cufftComplex* h_out;

      int x = dim[0];
      int y = dim[1];
      int z = dim[2];

      int size=x*y*z;

      h_in=(cufftReal*)malloc(sizeof(cufftReal)*size);
      h_out=(cufftComplex*)malloc(sizeof(cufftComplex)*size);

#if defined(__unix) || defined(__linux)

      if (time){
	gettimeofday(&end, NULL);
	endtime=end.tv_sec+(end.tv_usec/1000000.0);
	fprintf(stderr, "\n   mallocs took %.6f seconds.\n", endtime-starttime);
	gettimeofday(&start, NULL);
	starttime=start.tv_sec+start.tv_usec/1000000.0;
      }

#endif

      for(int i = 0; i < size; i++)
	{
	  h_in[i]=(cufftReal)input[2*i];
	}

#if defined(__unix) || defined(__linux)

      if (time){
	gettimeofday(&end, NULL);
	endtime=end.tv_sec+(end.tv_usec/1000000.0);
	fprintf(stderr, "\n   Population took %.6f seconds.\n", endtime-starttime);
	gettimeofday(&start, NULL);
	starttime=start.tv_sec+start.tv_usec/1000000.0;
      }

#endif

      unsigned int pad_length=size;
      //find the smallest power of 2 greater than or equal to pad_length
      pad_length-=1;
      pad_length|=(pad_length>>1);
      pad_length|=(pad_length>>2);
      pad_length|=(pad_length>>4);
      pad_length|=(pad_length>>8);
      pad_length|=(pad_length>>16);
      pad_length+=1;

      cudaMalloc((void**)&d_in, pad_length*sizeof(cufftReal)*2);

      cudaMemset(d_in, 0, pad_length*2);

      cudaMemcpy(d_in, h_in, sizeof(cufftReal)*size, cudaMemcpyHostToDevice);

#if defined(__unix) || defined(__linux)

      if (time){
	gettimeofday(&end, NULL);
	endtime=end.tv_sec+(end.tv_usec/1000000.0);
	fprintf(stderr, "\n   cudaMalloc/cudaMemset/cudaMemcpy took %.6f seconds.\n", endtime-starttime);
	gettimeofday(&start, NULL);
	starttime=start.tv_sec+start.tv_usec/1000000.0;
      }

#endif

      cufftHandle plan;

      cufftPlan3d(&plan, x, y, z, CUFFT_R2C);

      d_out=(cufftComplex*)d_in;

#if defined(__unix) || defined(__linux)

      if (time){
	gettimeofday(&end, NULL);
	endtime=end.tv_sec+(end.tv_usec/1000000.0);
	fprintf(stderr, "\n   Planning took %.6f seconds.\n", endtime-starttime);
	gettimeofday(&start, NULL);
	starttime=start.tv_sec+start.tv_usec/1000000.0;
      }

#endif

      cufftExecR2C(plan, d_in, d_out);

#if defined(__unix) || defined(__linux)

      if (time){
	gettimeofday(&end, NULL);
	endtime=end.tv_sec+(end.tv_usec/1000000.0);
	fprintf(stderr, "\n   Execution took %.6f seconds.\n", endtime-starttime);
	gettimeofday(&start, NULL);
	starttime=start.tv_sec+start.tv_usec/1000000.0;
      }

#endif

      cudaMemcpy(h_out, d_out, sizeof(cufftComplex)*size, cudaMemcpyDeviceToHost);

#if defined(__unix) || defined(__linux)

      if (time){
	gettimeofday(&end, NULL);
	endtime=end.tv_sec+(end.tv_usec/1000000.0);
	fprintf(stderr, "\n   cudaMemcpy took %.6f seconds.\n", endtime-starttime);
	gettimeofday(&start, NULL);
	starttime=start.tv_sec+start.tv_usec/1000000.0;
      }

#endif

      for(int i=0;i<size;i++)
	{
	  input[i*2] = h_out[i].x;
	  input[(i*2)+1] = h_out[i].y;
	}

#if defined(__unix) || defined(__linux)

      if (time){
	gettimeofday(&end, NULL);
	endtime=end.tv_sec+(end.tv_usec/1000000.0);
	fprintf(stderr, "\n   Copying results took %.6f seconds.\n", endtime-starttime);
	gettimeofday(&start, NULL);
	starttime=start.tv_sec+start.tv_usec/1000000.0;
      }

#endif

      cufftDestroy(plan);
      cudaFree(d_in);
      free(h_in);
      free(h_out);
    }
    else{

      cufftComplex* d_in;
      cufftReal* d_out;
      cufftComplex* h_in;
      cufftReal* h_out;

      int x = dim[0];
      int y = dim[1];
      int z = dim[2];

      int size=x*y*z;
      h_in=(cufftComplex*)malloc(sizeof(cufftComplex)*size);
      h_out=(cufftReal*)malloc(sizeof(cufftReal)*size);

#if defined(__unix) || defined(__linux)

      if (time){
	gettimeofday(&end, NULL);
	endtime=end.tv_sec+(end.tv_usec/1000000.0);
	fprintf(stderr, "\n   mallocs took %.6f seconds.\n", endtime-starttime);
	gettimeofday(&start, NULL);
	starttime=start.tv_sec+start.tv_usec/1000000.0;
      }

#endif

      for(int i = 0; i < size; i++)
	{
	  h_in[i].x=input[2*i];
	  h_in[i].y=input[(2*i)+1];
	}

#if defined(__unix) || defined(__linux)

      if (time){
	gettimeofday(&end, NULL);
	endtime=end.tv_sec+(end.tv_usec/1000000.0);
	fprintf(stderr, "\n   Population took %.6f seconds.\n", endtime-starttime);
	gettimeofday(&start, NULL);
	starttime=start.tv_sec+start.tv_usec/1000000.0;
      }

#endif

      unsigned int pad_length=size;
      //find the smallest power of 2 greater than or equal to pad_length
      pad_length-=1;
      pad_length|=(pad_length>>1);
      pad_length|=(pad_length>>2);
      pad_length|=(pad_length>>4);
      pad_length|=(pad_length>>8);
      pad_length|=(pad_length>>16);
      pad_length+=1;

      cudaMalloc((void**)&d_in, pad_length*sizeof(cufftComplex));

      cudaMemset(d_in, 0, pad_length*2);

      cudaMemcpy(d_in, h_in, sizeof(cufftComplex)*size, cudaMemcpyHostToDevice);

#if defined(__unix) || defined(__linux)

      if (time){
	gettimeofday(&end, NULL);
	endtime=end.tv_sec+(end.tv_usec/1000000.0);
	fprintf(stderr, "\n   cudaMalloc/cudaMemset/cudaMemcpy took %.6f seconds.\n", endtime-starttime);
	gettimeofday(&start, NULL);
	starttime=start.tv_sec+start.tv_usec/1000000.0;
      }

#endif

      cufftHandle plan;

      cufftPlan3d(&plan, x, y, z, CUFFT_C2R);

      d_out=(cufftReal*)d_in;

#if defined(__unix) || defined(__linux)

      if (time){
	gettimeofday(&end, NULL);
	endtime=end.tv_sec+(end.tv_usec/1000000.0);
	fprintf(stderr, "\n   Planning took %.6f seconds.\n", endtime-starttime);
	gettimeofday(&start, NULL);
	starttime=start.tv_sec+start.tv_usec/1000000.0;
      }

#endif

      cufftExecC2R(plan, d_in, d_out);

#if defined(__unix) || defined(__linux)

      if (time){
	gettimeofday(&end, NULL);
	endtime=end.tv_sec+(end.tv_usec/1000000.0);
	fprintf(stderr, "\n   Execution took %.6f seconds.\n", endtime-starttime);
	gettimeofday(&start, NULL);
	starttime=start.tv_sec+start.tv_usec/1000000.0;
      }

#endif

      cudaMemcpy(h_out, d_out, sizeof(cufftReal)*size, cudaMemcpyDeviceToHost);

#if defined(__unix) || defined(__linux)

      if (time){
	gettimeofday(&end, NULL);
	endtime=end.tv_sec+(end.tv_usec/1000000.0);
	fprintf(stderr, "\n   cudaMemcpy took %.6f seconds.\n", endtime-starttime);
	gettimeofday(&start, NULL);
	starttime=start.tv_sec+start.tv_usec/1000000.0;
      }

#endif

      int normalizer=size;

      for(int i=0;i<size;i++)
	{
	  input[2*i] = (float)h_out[i]/normalizer;
	}

#if defined(__unix) || defined(__linux)

      if (time){
	gettimeofday(&end, NULL);
	endtime=end.tv_sec+(end.tv_usec/1000000.0);
	fprintf(stderr, "\n   Copying results took %.6f seconds.\n", endtime-starttime);
	gettimeofday(&start, NULL);
	starttime=start.tv_sec+start.tv_usec/1000000.0;
      }

#endif

      cufftDestroy(plan);
      cudaFree(d_in);
      free(h_in);
      free(h_out);
    }
  }

#if defined(__unix) || defined(__linux)

  if (time){
    gettimeofday(&e, NULL);
    etime=e.tv_sec+e.tv_usec/1000000.0;
    fprintf(stderr, "\n  Full thing took %.6f seconds.\n", etime-stime);
  }

#endif

}
