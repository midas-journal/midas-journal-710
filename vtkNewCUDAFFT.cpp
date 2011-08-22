#include "vtkObjectFactory.h"
#include "vtkDataArray.h"
#include "vtkPointData.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkNewCUDAFFT.h"
#include "NewCUDAFFT.cu"

#if defined(__unix) || defined(__linux)
#include <sys/time.h>
#endif

#include <cufft.h>
#include <cuda.h>
#include <cuda_runtime.h>

extern "C" void NewCUDAFFT(float *input, int dim[3], int forward, int doComplex, int time);

vtkNewCUDAFFT* vtkNewCUDAFFT::New()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkNewCUDAFFT");
  if(ret)
    {
      return (vtkNewCUDAFFT*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkNewCUDAFFT;
}


vtkNewCUDAFFT::vtkNewCUDAFFT() 
{
  this->DebugMode=0;
  this->DoComplex=0;
  this->Forward=1;
  this->TimingMode=0;
  this->inData=NULL;
  this->outData=NULL;
}

vtkNewCUDAFFT::~vtkNewCUDAFFT() 
{

}


int vtkNewCUDAFFT::RequestInformation(vtkInformation *vtkNotUsed(request),  vtkInformationVector **inputVector, 
				      vtkInformationVector *outputVector)

{ 
  vtkInformation *outInfo = outputVector->GetInformationObject(0); 
  vtkDataObject::SetPointDataActiveScalarInfo(outInfo, VTK_DOUBLE, -1);
  vtkDataObject::SetPointDataActiveScalarInfo(outputVector->GetInformationObject(0), -1, 2);
  return 1;
}


// ---------------------------------------------------------------------------------------------------------------------------
//   Main Function
// ---------------------------------------------------------------------------------------------------------------------------
void vtkNewCUDAFFT::SimpleExecute(vtkImageData* input ,vtkImageData* output)
{

#if defined(__unix) || defined(__linux)  

  timeval start, end, s, e;
  double starttime, endtime, stime, etime;

  if (this->TimingMode){
    gettimeofday(&s, NULL);
    stime=s.tv_sec+s.tv_usec/1000000.0;
    gettimeofday(&start, NULL);
    starttime=start.tv_sec+start.tv_usec/1000000.0;
  }

#endif

  if (this->DebugMode)
    fprintf(stderr,"\nBeginning vtkNewCUDAFFT\n");

  vtkDataArray* inp=(this->inData==NULL)?(input->GetPointData()->GetScalars()):(this->inData);
  vtkDataArray* out=(this->outData==NULL)?(output->GetPointData()->GetScalars()):(this->outData);
  int nc=out->GetNumberOfComponents();
  for (int ia=0;ia<nc;ia++)
    out->FillComponent(ia,0.0);

  int dim[3]; input->GetDimensions(dim);
  int nt0=dim[0]*dim[1]*dim[2];
  float* inputdata=new float[nt0*2];

  if (this->DebugMode)
    fprintf(stderr,"nt0=%d nc1=%d dims=(%d,%d,%d)\n",nt0,1,dim[0],dim[1],dim[2]);

#if defined(__unix) || defined(__linux)

  if (this->TimingMode){
    gettimeofday(&end, NULL);
    endtime=end.tv_sec+(end.tv_usec/1000000.0);
    fprintf(stderr, "\n Getting information took %.6f seconds.\n", endtime-starttime);
    gettimeofday(&start, NULL);
    starttime=start.tv_sec+start.tv_usec/1000000.0;
  }

#endif
  
  if (this->DoComplex){
    if (this->Forward){
      // setting the real part first
      for (int i=0;i<nt0;i++)
	{
	  int index=i*2;
	  inputdata[index]=(float)inp->GetComponent(i,0); 	  
	  inputdata[index+1]=0.0f;
	}
    }
    else {
      for (int i=0;i<nt0;i++)
	{
	  int index=i*2;
	  inputdata[index]=(float)inp->GetComponent(i,0); 	  
	  inputdata[index+1]=(float)inp->GetComponent(i,1);
	}      
    }
  }
  else {
    if (this->Forward){
      //padding for CUFFT R2C
      int i;
      for (i=0; i<nt0; i++)
	inputdata[i]=(float)inp->GetComponent(i,0);
      for (; i<2*nt0; i++)
	inputdata[i]=0.0f;
    }
    else {
      for (int i=0;i<nt0;i++)
	{
	  int index=i*2;
	  inputdata[index]=(float)inp->GetComponent(i,0); 	  
	  inputdata[index+1]=(float)inp->GetComponent(i,1);
	}            
    }
  }
#if defined(__unix) || defined(__linux)

  if (this->TimingMode){
    gettimeofday(&end, NULL);
    endtime=end.tv_sec+(end.tv_usec/1000000.0);
    fprintf(stderr, "\n Populating took %.6f seconds.\n", endtime-starttime);
    gettimeofday(&start, NULL);
    starttime=start.tv_sec+start.tv_usec/1000000.0;
  }

#endif
  
  NewCUDAFFT(inputdata, dim, this->Forward, this->DoComplex, this->TimingMode);  
  
#if defined(__unix) || defined(__linux)

  if (this->TimingMode){
      gettimeofday(&end, NULL);
      endtime=end.tv_sec+(end.tv_usec/1000000.0);
      fprintf(stderr, "\n NewCUDAFFT took %.6f seconds.\n", endtime-starttime);
      gettimeofday(&start, NULL);
      starttime=start.tv_sec+start.tv_usec/1000000.0;
  }

#endif
  
  if (this->DoComplex){
    if (this->Forward){
      
      for (int i=0;i<nt0;i++)
	{
	  int index=2*i;
	  out->SetComponent(i,0,inputdata[index]);
	  out->SetComponent(i,1,inputdata[index+1]);
	}
    }
    else {
      for (int i=0; i<nt0; i++){
	out->SetComponent(i,0,inputdata[2*i]);
	//      out->SetComponent(i,1,0.0f);
      }
    }
  }
  else {
    //problem: the symmetries CUFFT expects you to know to use R2C/C2R 3d FFT/RFFT output are (at least to me, with my limited background) nontrivial
    //the following data copying functions are WRONG in that they simply copy non-redundant coefficients rather than the actual FFT/RFFT results (that can be calculated using the CUFFT output)
    //if you know more than me about this (which you probably do), fix this up
    //otherwise, don't have DoComplex=0!!!
    if (this->Forward){      
      for (int i=0;i<nt0;i++)
	{
	  int index=2*i;
	  out->SetComponent(i,0,inputdata[index]);
	  out->SetComponent(i,1,inputdata[index+1]);
	}
    }
    else {
      for (int i=0; i<nt0; i++){
	out->SetComponent(i,0,inputdata[i]);
	//      out->SetComponent(i,1,0.0f);
      }
    }    
  }
  
#if defined(__unix) || defined(__linux)

  if (this->TimingMode){
    gettimeofday(&end, NULL);
    endtime=end.tv_sec+(end.tv_usec/1000000.0);
    fprintf(stderr, "\n Copying answer took %.6f seconds.\n", endtime-starttime);
    gettimeofday(&start, NULL);
    starttime=start.tv_sec+start.tv_usec/1000000.0;
  }

#endif

  delete [] inputdata;
  
#if defined(__unix) || defined(__linux)

  if (this->TimingMode){
    gettimeofday(&e, NULL);
    etime=e.tv_sec+e.tv_usec/1000000.0;
    fprintf(stderr, "\n-Full FFT took %.6f seconds.\n", etime-stime);
  }
  
#endif

}
