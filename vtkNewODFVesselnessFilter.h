#ifndef __vtkNewODFVesselnessFilter_h
#define __vtkNewODFVesselnessFilter_h

#include "vtkSimpleImageToImageFilter.h"
#include "vtkImageData.h"

class vtkNewODFVesselnessFilter : public vtkSimpleImageToImageFilter
{
public:
  static vtkNewODFVesselnessFilter *New();
  vtkTypeMacro(vtkNewODFVesselnessFilter,vtkSimpleImageToImageFilter);

  // Description:
  // Beta Parameter for Distribution
  vtkSetMacro(Beta,double);
  vtkGetMacro(Beta,double);
  
  // Description
  // Tau Parameter for Entropy
  vtkSetMacro(Tau,double);
  vtkGetMacro(Tau,double);
  
  // Description:
  // Generate ODF Image
  vtkSetClampMacro(OutputODF,int,0,1);
  vtkGetMacro(OutputODF,int);

  // Description:
  // Print timing data
  vtkSetClampMacro(TimingMode,int,0,1);
  vtkGetMacro(TimingMode,int);  

  // Description:
  // Force CPU compution
  vtkSetClampMacro(ForceCPU,int,0,1);
  vtkGetMacro(ForceCPU,int);  

  // Description:
  // ODF Image if stored
  vtkGetObjectMacro(ODFImage,vtkImageData);


protected:

  vtkNewODFVesselnessFilter();
  virtual ~vtkNewODFVesselnessFilter();
  vtkNewODFVesselnessFilter(const vtkNewODFVesselnessFilter&) {};
  void operator=(const vtkNewODFVesselnessFilter&) {};

  // Description:
  // Basic Input Outputs
  virtual int RequestInformation(vtkInformation *vtkNotUsed(request),  
				 vtkInformationVector **inputVector, 
				 vtkInformationVector *outputVector);
  //  virtual void ExecuteInformation();
  virtual void SimpleExecute(vtkImageData* inp,vtkImageData* out);

  // Description:
  // Data
  double      Beta;
  double      Tau;
  vtkImageData* ODFImage;
  int         OutputODF;
  int         TimingMode;
  int         ForceCPU;

  // Description:
  virtual void ComputeEntropyMeasure(vtkDataArray* inp,vtkDataArray* out,vtkDataArray* odf, vtkImageData* input, int forcecpu);
};

#endif
