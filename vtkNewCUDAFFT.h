#ifndef __vtkNewCUDAFFT_h
#define __vtkNewCUDAFFT_h

#include "vtkSimpleImageToImageFilter.h"
#include "vtkImageData.h"
#include "vtkDataArray.h"

class vtkNewCUDAFFT : public vtkSimpleImageToImageFilter {
 public:
  static vtkNewCUDAFFT *New();
  vtkTypeMacro(vtkNewCUDAFFT,vtkSimpleImageToImageFilter);

  vtkGetMacro(DebugMode,int);
  vtkSetClampMacro(DebugMode,int,0,1);

  vtkGetMacro(DoComplex,int);
  vtkSetClampMacro(DoComplex,int,0,1);
  
  vtkGetMacro(Forward,int);
  vtkSetClampMacro(Forward,int,0,1);  

  vtkGetMacro(TimingMode,int);
  vtkSetClampMacro(TimingMode,int,0,1);

  vtkSetObjectMacro(inData,vtkDataArray);
  vtkGetObjectMacro(inData,vtkDataArray);

  vtkSetObjectMacro(outData,vtkDataArray);
  vtkGetObjectMacro(outData,vtkDataArray);

 protected:
  vtkNewCUDAFFT();
  virtual ~vtkNewCUDAFFT();
  
  virtual void SimpleExecute(vtkImageData* input,vtkImageData* output);
  virtual int RequestInformation(vtkInformation *vtkNotUsed(request),  
				 vtkInformationVector **inputVector, 
				 vtkInformationVector *outputVector);

  int DebugMode;
  int DoComplex;
  int Forward;
  int TimingMode;
  //BTX
  vtkDataArray *inData, *outData;
  //ETX

 private:  
  vtkNewCUDAFFT(const vtkNewCUDAFFT& src){};
  vtkNewCUDAFFT& operator=(const vtkNewCUDAFFT& rhs){};
};

#endif /* VTKNEWCUDAFFT_H_ */
