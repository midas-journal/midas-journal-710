#ifndef __vtkNewImageConvolution_h
#define __vtkNewImageConvolution_h

#include "vtkSimpleImageToImageFilter.h"
#include "vtkImageData.h"
#include "vtkPoints.h"
#include "vtkFloatArray.h"

class vtkNewImageConvolution : public vtkSimpleImageToImageFilter
{
 public:
  static vtkNewImageConvolution *New();
  vtkTypeMacro(vtkNewImageConvolution,vtkSimpleImageToImageFilter);
  
  // Description:
  // DistanceMap -- distance from traced structure
  vtkSetObjectMacro(FilterBank,vtkImageData);
  vtkGetObjectMacro(FilterBank,vtkImageData);

  // Description:
  // Mode
  vtkGetMacro(Mode,int);
  vtkSetClampMacro(Mode,int,0,2);
  virtual void SetModeToConvolve()  { this->SetMode(0); }
  virtual void SetModeToDeviation() { this->SetMode(1); }
  virtual void SetModeToBoth()      { this->SetMode(2); }

  // Description:
  // Second Output is Mean Image in case when Mode = 2
  vtkGetObjectMacro(SecondOutput,vtkImageData);

  // Description:
  // Precision
  vtkSetClampMacro(DoublePrecision,int,0,1);
  vtkGetMacro(DoublePrecision,int);

  // Description:
  // Use CPU/GPU
  vtkSetClampMacro(ForceCPU,int,0,1);
  vtkGetMacro(ForceCPU,int);

  // Description:
  // Use Complex to Complex mappings
  vtkSetClampMacro(DoComplexFFTs,int,0,1);
  vtkGetMacro(DoComplexFFTs,int);

  // Description:
  // Output timings
  vtkSetClampMacro(TimingMode,int,0,1);
  vtkGetMacro(TimingMode,int);

    // XQ's version:
  static vtkImageData* qVesselFilter(double anglespacing,int radius,double sigma_r=3.0,double sigma=2.0);
 protected:

  vtkNewImageConvolution();
  virtual ~vtkNewImageConvolution();
  vtkNewImageConvolution(const vtkNewImageConvolution&) {};
  void operator=(const vtkNewImageConvolution&) {};

  // Description:
  // Basic Input Outputs
  // ----------------------------------------------------------------------------
  virtual int RequestInformation(vtkInformation *vtkNotUsed(request),  
				 vtkInformationVector **inputVector, 
				 vtkInformationVector *outputVector);
  virtual void SimpleExecute(vtkImageData* inp,vtkImageData* out);

  // Description:
  // Data
  // ----------------------------------------------------------------------------
  vtkImageData* FilterBank;
  vtkImageData* SecondOutput;
  int           Mode;
  int           DoublePrecision;
  int           ForceCPU;
  int           DoComplexFFTs;
  int           TimingMode;

  // Description:
  // Normalize 'new' sphere filters.
  virtual void normalizeFilters(vtkImageData* FILTERS);
  // Description:
  // Basic Operations for Convolutions in Fourier Space
  // ----------------------------------------------------------------------------
  virtual vtkImageData* PadImage(vtkImageData* img,int paddim[3]);
  virtual vtkImageData* FFTImage(vtkImageData* img,int frame,int paddim[3],int dosquare, vtkDataArray* data=NULL);
  virtual vtkImageData* DoFourierConvolution(vtkImageData* img1,vtkImageData* img2,int origdim[3],double origori[3], int DoSquare);
};

#endif
