#include "vtkObjectFactory.h"
#include "vtkNewODFVesselnessFilter.h"
#include "vtkImageReslice.h"
#include "vtkImageExtractComponents.h"
#include "vtkImageFFT.h"
#include "vtkImageRFFT.h"
#include "vtkImageCast.h"
#include "vtkImageMagnitude.h"
#include "vtkImageMathematics.h"
#include "vtkPointData.h"
#include "vtkDataArray.h"
#include "vtkImageShiftScale.h"
#include "vtkMath.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkStreamingDemandDrivenPipeline.h"
#include <assert.h>

#if defined(__unix) || defined(__linux)
#include <sys/time.h>
#endif

//------------------------------------------------------------------------------
extern "C" void NewCUDAEntropy(float* inputdata, float beta, float tau, int dim[3], int nc, int time);

vtkNewODFVesselnessFilter* vtkNewODFVesselnessFilter::New()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkNewODFVesselnessFilter");
  if(ret)
    {
    return (vtkNewODFVesselnessFilter*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkNewODFVesselnessFilter;
}

// Construct object with no children.
vtkNewODFVesselnessFilter::vtkNewODFVesselnessFilter()
{
  this->Tau=1.0;
  this->Beta=0.01;
  this->OutputODF=0;
  this->ODFImage=NULL;
  this->Debug=0;
  this->TimingMode=0;
  this->ForceCPU=0;
}
// ----------------------------------------------------------------------------
vtkNewODFVesselnessFilter::~vtkNewODFVesselnessFilter()
{
  if (this->ODFImage!=NULL)
    this->ODFImage->Delete();
}
int vtkNewODFVesselnessFilter::RequestInformation(vtkInformation *vtkNotUsed(request),  vtkInformationVector **inputVector, 
						   vtkInformationVector *outputVector)
{
  vtkInformation* outInfo=outputVector->GetInformationObject(0);
  vtkDataObject::SetPointDataActiveScalarInfo(outInfo, VTK_FLOAT, -1);
  vtkDataObject::SetPointDataActiveScalarInfo(outInfo, -1, 1);
  if (this->Debug)
    fprintf(stderr, "In Req. Info, nc=%d\n", 1);
  return 1;
}

void vtkNewODFVesselnessFilter::SimpleExecute(vtkImageData* input,vtkImageData* output)
{

#if defined(__unix) || defined(__linux)

  timeval start, end;
  double starttime, endtime;

#endif

  if (input==NULL)
    {
      vtkErrorMacro(<<"Bad Input to vtkNewODFVesselnessFilter");
      return;
    }
  
  vtkDataArray* inp=input->GetPointData()->GetScalars();
  vtkDataArray* out=output->GetPointData()->GetScalars();

  if (this->ODFImage!=NULL)
    {
      this->ODFImage->Delete();
      this->ODFImage=NULL;
    }

  vtkDataArray* odf=NULL;
  if (this->OutputODF)
    {
      this->ODFImage=vtkImageData::New();
      this->ODFImage->CopyStructure(input);
      this->ODFImage->SetScalarTypeToFloat();
      this->ODFImage->AllocateScalars();
      odf=this->ODFImage->GetPointData()->GetScalars();
    }

#if defined(__unix) || defined(__linux)

  if (this->TimingMode){
    gettimeofday(&start, NULL);
    starttime=start.tv_sec+(start.tv_usec/1000000.0);
  }

#endif

  this->ComputeEntropyMeasure(inp,out,odf, input, this->ForceCPU);

#if defined(__unix) || defined(__linux)

  if (this->TimingMode){
    gettimeofday(&end, NULL);
    endtime=end.tv_sec+(end.tv_usec/1000000.0);
    fprintf(stderr, "\nEntropy calculations took %.6f seconds.\n", endtime-starttime);
  }

#endif

  if (this->Debug)
    this->UpdateProgress(1.0);
}

void vtkNewODFVesselnessFilter::ComputeEntropyMeasure(vtkDataArray* inp,vtkDataArray* out,vtkDataArray* odf, vtkImageData* input, int forcecpu)
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

  //if (forcecpu || odf!=NULL){
    int nc=inp->GetNumberOfComponents();
    int nt=inp->GetNumberOfTuples();
    double* p=new double[nc];
    
    int tenth=nt/10;
    int count=0;
    
    for (int voxel=0;voxel<nt;voxel++)
      {
	double sum=0.0;
	for (int comp=0;comp<nc;comp++)
	  {
	    double v=inp->GetComponent(voxel,comp);
	    p[comp]=exp(-this->Beta*v);
	    sum+=p[comp];
	  }
	double entropy=0.0;
	for (int comp=0;comp<nc;comp++)
	  {
	    if (sum>1e-6)
	      p[comp]/=sum;
	    else
	      p[comp]=0.0;
	    
	    if (odf!=NULL)
	      odf->SetComponent(voxel,comp,p[comp]);
	  if (p[comp]>1e-6)
	    entropy+= p[comp]*log(p[comp]);
	  }
	double vess=exp(this->Tau*entropy);
	out->SetComponent(voxel,0,vess);
	++count;
	if (count==tenth && this->Debug)
	  {
	    this->UpdateProgress(this->GetProgress()+0.1);
	    count=0;
	  }
      }
    delete [] p;
  

}

