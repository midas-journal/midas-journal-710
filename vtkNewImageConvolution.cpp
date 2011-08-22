#include "vtkObjectFactory.h"
#include "vtkNewImageConvolution.h"
#include "vtkImageReslice.h"
#include "vtkImageExtractComponents.h"
#include "vtkImageFFT.h"
#include "vtkImageRFFT.h"
#include "vtkImageFFT.h"
#include "vtkImageRFFT.h"

#include "vtkImageCast.h"
#include "vtkImageMagnitude.h"
#include "vtkImageMathematics.h"
#include "vtkPointData.h"
#include "vtkDataArray.h"
#include "vtkImageShiftScale.h"
#include "vtkMath.h"
#include "vtkPoints.h"
#include "vtkPolyData.h"
#include "vtkPolyDataWriter.h"
#include "vtkCellArray.h"
#include "vtkSphereSource.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkStreamingDemandDrivenPipeline.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "pxisinf.h"

#if defined(__unix) || defined(__linux)
#include <sys/time.h>
#endif

#include <math.h>

// CUDA include files
#include "vtkNewCUDAFFT.h"

vtkNewImageConvolution* vtkNewImageConvolution::New()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkNewImageConvolution");
  if(ret)
    {
    return (vtkNewImageConvolution*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkNewImageConvolution;
}

// Construct object with no children.
vtkNewImageConvolution::vtkNewImageConvolution()
{
  this->Mode=1;
  this->DoublePrecision=1;
  this->FilterBank=NULL;
  this->SecondOutput=NULL;
  this->ForceCPU=0;
  this->DoComplexFFTs=0;
  this->TimingMode=0;
}
// ----------------------------------------------------------------------------
vtkNewImageConvolution::~vtkNewImageConvolution()
{
  this->SetFilterBank(NULL);
  if (this->SecondOutput!=NULL)
    this->SecondOutput->Delete();
}

// ----------------------------------------------------------------------------
int vtkNewImageConvolution::RequestInformation(vtkInformation *vtkNotUsed(request),  vtkInformationVector **inputVector,
						     vtkInformationVector *outputVector)

{
  vtkInformation *outInfo = outputVector->GetInformationObject(0);
  vtkDataObject::SetPointDataActiveScalarInfo(outInfo, VTK_FLOAT, -1);
  vtkDataObject::SetPointDataActiveScalarInfo(outputVector->GetInformationObject(0), -1, 1);

  if (this->FilterBank==NULL)
    {
      vtkErrorMacro(<<"No Filter Bank Image Set\n");
      return 1;
    }
  int nc=this->FilterBank->GetNumberOfScalarComponents();
  vtkDataObject::SetPointDataActiveScalarInfo(outputVector->GetInformationObject(0), -1, nc);
  return 1;
}
// -------------------------------------------------------------------------------------------------------------
void vtkNewImageConvolution::SimpleExecute(vtkImageData* input,vtkImageData* output)
{

#if defined(__unix) || defined(__linux)

  timeval start, end;
  double starttime, endtime;

#endif

  if (this->Debug)
    fprintf(stderr, "In simple execute of FOURIER CONVOLUTION ..........................\n");
  if (input==NULL    || this->FilterBank ==NULL )
    {
      vtkErrorMacro(<<"Bad Inputs to vtkNewImageConvolution");
      return;
    }

  double tmp[3]; input->GetOrigin(tmp);
  if (this->Debug)
    fprintf(stderr,"Input Origin = %.2f,%.2f,%.2f\n",tmp[0],tmp[1],tmp[2]);

#if defined(__unix) || defined(__linux)

  if (this->TimingMode){
    gettimeofday(&start, NULL);
    starttime=start.tv_sec+(start.tv_usec/1000000.0);
  }

#endif

  if (this->Debug)
    fprintf(stderr, "Normalizing filters...\n");
  this->normalizeFilters(this->FilterBank);
  if (this->Debug)
    fprintf(stderr, "Normalized.\n");

#if defined(__unix) || defined(__linux)

  if (this->TimingMode){
    gettimeofday(&end, NULL);
    endtime=end.tv_sec+(end.tv_usec/1000000.0);
    fprintf(stderr, "\nNormalization took %.6f seconds.\n", endtime-starttime);
  }

#endif


  int nc=this->FilterBank->GetNumberOfScalarComponents();
  int original_dim[3];    input->GetDimensions(original_dim);
  double offset_origin[3]; input->GetOrigin(offset_origin);
  double spa[3]; input->GetSpacing(spa);
  int fdim[3]; this->FilterBank->GetDimensions(fdim);

  if (this->SecondOutput!=NULL)
    this->SecondOutput->Delete();
  this->SecondOutput=NULL;

  if (this->Mode==2)
    {
      this->SecondOutput=vtkImageData::New();
      this->SecondOutput->CopyStructure(output);
      this->SecondOutput->SetNumberOfScalarComponents(output->GetNumberOfScalarComponents());
      this->SecondOutput->AllocateScalars();
      if (this->Debug)
	fprintf(stderr, "fb nc: %d output nc:%d secondoutput nc: %d\n", nc, output->GetNumberOfScalarComponents(), this->SecondOutput->GetNumberOfScalarComponents());
    }


  int paddim[3];
  for (int ia=0;ia<=2;ia++)
    {
      paddim[ia]=original_dim[ia]+fdim[ia]+1;
      offset_origin[ia]+=0.5*spa[ia]*double(fdim[ia]-1);
    }

  if (this->Debug)
    {
      fprintf(stderr,"Beginning Image Fourier Convolution: Mode = %d , Double Precision =%d\n",
	      this->Mode,this->DoublePrecision);
      fprintf(stderr,"Input dims = (%d,%d,%d) filter=(%d,%d,%d) nc=%d padded=(%d,%d,%d) ori=(%.1f,%.1f,%.1f)\n",
	      original_dim[0],   original_dim[1],   original_dim[2],
	      fdim[0],  fdim[1],  fdim[2],nc,
	      paddim[0],paddim[1],paddim[2],
	      offset_origin[0],offset_origin[1],offset_origin[2]);
    }

  int odim[3]; output->GetDimensions(odim);
  int onc=output->GetNumberOfScalarComponents();
  if (this->Debug)
    fprintf(stderr,"Output = (%d,%d,%d, %d) \n",
	    odim[0],odim[1],odim[2],onc);


  // Normalize Image between 0 and 1
  // --------------------------------
  double range[2]; input->GetPointData()->GetScalars()->GetRange(range);
  double sc=1.0/(range[1]-range[0]);

  if (this->Debug)
    fprintf(stderr,"Range = %f:%f, scale=%f\n",range[0],range[1],sc);

  vtkImageCast* cast=vtkImageCast::New();
  cast->SetInput(input);
  cast->SetOutputScalarTypeToFloat();
  cast->GetOutput()->SetDimensions(input->GetDimensions());
  cast->Update();

  vtkImageData* normalized=vtkImageData::New();
  if (this->Mode>=1)
    {
      vtkImageShiftScale* shiftscale=vtkImageShiftScale::New();
      shiftscale->SetInput(cast->GetOutput());
      shiftscale->SetShift(-range[0]);
      shiftscale->SetScale(sc);
      shiftscale->Update();
      normalized->ShallowCopy(shiftscale->GetOutput());
      shiftscale->Delete();
    }
  else
    {
      cast->Update();
      normalized->ShallowCopy(cast->GetOutput());
    }
  cast->Delete();

  // --------------------------------

#if defined(__unix) || defined(__linux)

  if (this->TimingMode){
    gettimeofday(&start, NULL);
    starttime=start.tv_sec+(start.tv_usec/1000000.0);
  }

#endif

  vtkDataArray *inp=normalized->GetPointData()->GetScalars();

  vtkImageData *FI=NULL, *FI2=NULL;
  FI=this->FFTImage(normalized, 0, paddim, 0, inp);//FI is now FFT(I)

#if defined(__unix) || defined(__linux)

  if (this->TimingMode){
    gettimeofday(&end, NULL);
    endtime=end.tv_sec+(end.tv_usec/1000000.0);
    fprintf(stderr, "\nFFT(input) took %.6f seconds.\n", endtime-starttime);
  }

#endif

  if (this->Mode==1 || this->Mode==2)
    {
  FI2=this->FFTImage(normalized, 0, paddim, 1, inp);//FI2 is now FFT^2(I)

#if defined(__unix) || defined(__linux)

      if (this->TimingMode){
	gettimeofday(&start, NULL);
	starttime=start.tv_sec+(start.tv_usec/1000000.0);
	fprintf(stderr, "\nSecond FFT(input) [square mode] took %.6f seconds.\n", starttime-endtime);
      }

#endif

    }

  vtkDataArray *out=output->GetPointData()->GetScalars();

  vtkDataArray *out2=NULL;
  if (this->SecondOutput!=NULL)
    {
      out2=this->SecondOutput->GetPointData()->GetScalars();
      out2->SetNumberOfComponents(output->GetNumberOfScalarComponents());
    }

  //  out->SetNumberOfComponents(output->GetNumberOfScalarComponents());


  //  fprintf(stderr, "outnc: %d, nc2: %d \n", out->GetNumberOfComponents(), out2->GetNumberOfComponents());

  int nt=out->GetNumberOfTuples();

  for (int frame=0;frame<nc;frame++)
  {
    out->FillComponent(frame,0.0);
    if (out2!=NULL)
      out2->FillComponent(frame,0.0);
  }

  int debugv=original_dim[0]/2+original_dim[0]*(original_dim[1]/2)+(original_dim[0]*original_dim[1])*original_dim[2]/2;

  for (int frame=0;frame<nc;frame++)
  {
	  if (this->Debug)
		  fprintf(stderr,"\n Beginning Filter %d/%d\n",frame+1,nc);
      this->UpdateProgress(double(frame)/double(nc));

#if defined(__unix) || defined(__linux)

      if (this->TimingMode){
    	  gettimeofday(&start, NULL);
    	  starttime=start.tv_sec+(start.tv_usec/1000000.0);
      }

#endif

      vtkImageData* Bi = FFTImage(this->FilterBank,frame,paddim,0);//Bi is the FFT of the i-th wedge

#if defined(__unix) || defined(__linux)

      if (this->TimingMode){
	gettimeofday(&end, NULL);
	endtime=end.tv_sec+(end.tv_usec/1000000.0);
	fprintf(stderr, "\nFFT(Bi) took %.6f seconds.\n", endtime-starttime);
      }

#endif


    vtkImageData* IxBi=this->DoFourierConvolution(FI,Bi,original_dim,offset_origin,0);//IxBi is the convolution of I and the i-th wedge--i.e. RFFT(FFT(I)*FFT(wedge_i))

#if defined(__unix) || defined(__linux)

      if (this->TimingMode){
	gettimeofday(&start, NULL);
	starttime=start.tv_sec+(start.tv_usec/1000000.0);
	fprintf(stderr, "\nConvolution of I and Bi took %.6f seconds.\n", starttime-endtime);
      }

#endif

      vtkDataArray *data=IxBi->GetPointData()->GetScalars();

      if (this->Mode==1 || this->Mode==2)
      {
  	  vtkImageData* I2xBi=this->DoFourierConvolution(FI2,Bi,original_dim,offset_origin,1);//similarly with I2xBi, only as a function of I^2

#if defined(__unix) || defined(__linux)

    	  if (this->TimingMode){
    		  gettimeofday(&end, NULL);
    		  endtime=end.tv_sec+(end.tv_usec/1000000.0);
    		  fprintf(stderr, "\nConvolution of Bi and I^2 took %.6f seconds.\n", endtime-starttime);
    	  }

#endif

   	  vtkDataArray *data2=I2xBi->GetPointData()->GetScalars();

   	  for (int j=0;j<nt;j++)//compute Dev per our previously derived formula
   	  {
   		  double lm=inp->GetComponent(j,0);
   		  double vl=data2->GetComponent(j,0)-2.0*lm*data->GetComponent(j,0)+lm*lm;
    		  if (isnan(vl) || isinf(vl))
    			  vl=0.0;
   		  out->SetComponent(j,frame,vl);

    		  if (j==debugv && this->Debug)
    			  fprintf(stderr,"Image Center vals = image=%f,  I2xBi=%f, IxBI=%f out=%f\n",
    					  lm,
    					  data2->GetComponent(j,0),
    					  data->GetComponent(j,0),
    					  out->GetComponent(j,frame));
   	  }
	  I2xBi->Delete();
	  if (this->Mode==2)
	  {
		  if (this->Debug)
			  fprintf(stderr,"Mode=2, frame=%d\n",frame);
		  out2->CopyComponent(frame,data,0);
	  }
      }
      else
      {
    	  if (this->Debug)
    		  fprintf(stderr,"Mode=0, frame=%d\n",frame);
    	  out->CopyComponent(frame,data,0);
      }
      //      fprintf(stderr,"Cleaning up 1\n");
      IxBi->Delete();
      Bi->Delete();
  }

  //  fprintf(stderr,"Cleaning up 2\n");
  FI->Delete();


  //  fprintf(stderr,"Cleaning up 3\n");
  if (FI2!=NULL)
    FI2->Delete();

  //  fprintf(stderr,"Cleaning up 4\n");
  normalized->Delete();

  //  fprintf(stderr,"Cleaning up 5\n");

  //fprintf(stderr,"nc = %d,  nc2=%d, \n", output->GetNumberOfScalarComponents(), this->SecondOutput->GetNumberOfScalarComponents());
  this->UpdateProgress(1.0);
}
// -------------------------------------------------------------------------------------------------------------
vtkImageData* vtkNewImageConvolution::PadImage(vtkImageData* img,int paddim[3])
{
  //  static int count=1;

  int dim[3];
  if (this->Debug)
    img->GetDimensions(dim);
  double ori[3];
  if (this->Debug)
    img->GetOrigin(ori);
  double spa[3];
  if (this->Debug)
    img->GetSpacing(spa);
  if (this->Debug)
    fprintf(stderr,"\t Padding = (%d,%d,%d) ori=(%.1f,%.1f,%.1f) spa=(%.1f,%.1f,%.1f)\n",
	    dim[0],	  dim[1],	  dim[2],
	    ori[0],	  ori[1],	  ori[2],
	    spa[0],	  spa[1],	  spa[2]);



  vtkImageReslice* resl=vtkImageReslice::New();
  resl->SetInput(img);
  resl->SetOutputExtent(0,paddim[0]-1,0,paddim[1]-1,0,paddim[2]-1);
  resl->SetOutputOrigin((this->Debug)?ori:img->GetOrigin());
  resl->SetOutputSpacing((this->Debug)?spa:img->GetSpacing());
  resl->SetInterpolationMode(0);
  resl->SetBackgroundLevel(0.0);
  resl->Update();

  vtkImageData* out=vtkImageData::New();
  out->ShallowCopy(resl->GetOutput());
  resl->Delete();

  if (this->Debug){
    out->GetDimensions(dim);
    out->GetOrigin(ori);
    out->GetSpacing(spa);
  }
  if (this->Debug)
    {
      fprintf(stderr,"\t Padding Done = (%d,%d,%d) ori=(%.1f,%.1f,%.1f) spa=(%.1f,%.1f,%.1f)\n",
	      dim[0],	  dim[1],	  dim[2],
	      ori[0],	  ori[1],	  ori[2],
	      spa[0],	  spa[1],	  spa[2]);


      /*      char line[100];
      sprintf(line,"padded_%d.nii.gz",count); ++count;
      vtkpxUtil::SaveAnalyze(line,out,9);
      fprintf(stderr,"padded image saved in %s\n",line);*/
    }
  return out;
}
// -------------------------------------------------------------------------------------------------------------
vtkImageData* vtkNewImageConvolution::FFTImage(vtkImageData* img,int frame,int paddim[3],int dosquare, vtkDataArray* inData)
{
  int dim[3];
  if (this->Debug)
    img->GetDimensions(dim);
  int nc;
  if (this->Debug)
    img->GetNumberOfScalarComponents();
  if (this->Debug)
    fprintf(stderr,"Performing FFT (%d x %d x %d, %d) --> (%d x %d x %d at %d) dosquare=%d\n",
	    dim[0],dim[1],dim[2],nc,
	    paddim[0],paddim[1],paddim[2],
	    frame,dosquare);

  vtkImageData* tmp=NULL;
  if (img->GetNumberOfScalarComponents()>1)
    {
      vtkImageExtractComponents* comp=vtkImageExtractComponents::New();
      comp->SetInput(img);
      comp->SetComponents(frame);
      if (this->Debug)
	fprintf(stderr,"\t\t Extracting frame=%d\n",frame);
      comp->Update();
      tmp=this->PadImage(comp->GetOutput(),paddim);
      comp->Delete();
    }
  else
    {
     tmp=this->PadImage(img,paddim);
    }


  vtkImageMathematics* math=vtkImageMathematics::New();

  int usegpu=this->ForceCPU<=0?1:0;

  vtkImageAlgorithm* fft=NULL;
  if (usegpu)
    {
      if (this->Debug)
    	  fprintf(stderr,"\nUsing CUDA\n");
  vtkNewCUDAFFT* fft2=vtkNewCUDAFFT::New();//implement CUDA's FFT/RFFT functionality
      fft2->SetDebugMode(this->Debug);
  fft2->SetForward(1);//set to FFT
      fft2->SetTimingMode(this->TimingMode);
      fft2->SetDoComplex(this->DoComplexFFTs);
      //      if (inData!=NULL)
      //	fft2->SetinData(inData);
  fft=fft2;
    }
  else
    {
      if (this->Debug)
	fprintf(stderr,"\nUsing cpu\n");
      vtkImageFourierFilter* fft2=vtkImageFFT::New();
      fft2->SetDimensionality(3);
      fft=fft2;
    }

  if (dosquare)
    {
      math->SetInput1(tmp);
      math->SetOperationToSquare();
      math->Update();
      fft->SetInput(math->GetOutput());
    }
  else
    {
      fft->SetInput(tmp);
    }

  tmp->Delete();

  if (this->Debug)
    fprintf(stderr, "\nComputing FFT\n");

  fft->Update();

  vtkImageData* out=vtkImageData::New();
  out->ShallowCopy(fft->GetOutput());

  double range1[2];
  if (this->Debug)
    out->GetPointData()->GetScalars()->GetRange(range1,0);
  double range2[2];
  if (this->Debug)
    out->GetPointData()->GetScalars()->GetRange(range2,1);
  if (this->Debug)
    fprintf(stderr,"\t FFT RANGE=%f:%f\n",range1[0],range1[1],range2[0],range2[1]);
  if (this->Debug)
    out->GetDimensions(dim);
  if (this->Debug)
    fprintf(stderr,"\t FFT Done (%dx%dx%d,%d)\n",
	    dim[0],dim[1],dim[2],out->GetNumberOfScalarComponents());



  fft->Delete();
  math->Delete();

  return out;
}
// -------------------------------------------------------------------------------------------------------------
vtkImageData* vtkNewImageConvolution::DoFourierConvolution(vtkImageData* img1,vtkImageData* img2,int origdim[3],double origori[3],int dosquare)
{

#if defined(__unix) || defined(__linux)

  timeval start, end;
  double starttime, endtime;

#endif

  //  static int count=1;
  int dim1[3];
  if (this->Debug)
    img1->GetDimensions(dim1);
  double ori1[3];
  if (this->Debug)
    img1->GetOrigin(ori1);

  // Force Origin and Spacing the same -- not that it should matter
  img2->SetSpacing(img1->GetSpacing());
  img2->SetOrigin(img1->GetOrigin());

  int dim2[3];
  if (this->Debug)
    img2->GetDimensions(dim2);
  double ori2[3];
  if (this->Debug)
    img2->GetOrigin(ori2);

  int nc1, nc2;
  if (this->Debug){
    nc1=img1->GetNumberOfScalarComponents();
    nc2=img2->GetNumberOfScalarComponents();
  }
  if (this->Debug)
    fprintf(stderr,"Performing F-Conv (%d x %d x %d, %d) and (%d x %d x %d, %d) ori1=(%.1f,%.1f,%.1f), ori2=(%.1f,%.1f,%.1f)--> (%d x %d x %d ), oriout=(%.1f,%.1f,%.1f)\n",
	    dim1[0],dim1[1],dim1[2],nc1,
	    dim2[0],dim2[1],dim2[2],nc2,
	    ori1[0],ori1[1],ori1[2],
	    ori2[0],ori2[1],ori2[2],
	    origdim[0],origdim[1],origdim[2],
	    origori[0],origori[1],origori[2]);


#if defined(__unix) || defined(__linux)

  if (this->TimingMode){
    gettimeofday(&start, NULL);
    starttime=start.tv_sec+(start.tv_usec/1000000.0);
  }

#endif

  vtkImageMathematics* math=vtkImageMathematics::New();
  math->DebugOff();
  math->SetInput1(img1);
  math->SetInput2(img2);
  math->SetOperationToComplexMultiply();
  //  math->Update();


#if defined(__unix) || defined(__linux)

  if (this->TimingMode){
    gettimeofday(&end, NULL);
    endtime=end.tv_sec+(end.tv_usec/1000000.0);
    fprintf(stderr, "\n  Complex multiplication took %.6f seconds.\n", endtime-starttime);
  }

#endif

  int usegpu=this->ForceCPU<=0?1:0;

  vtkImageAlgorithm* rfft=NULL;
  if (usegpu)
    {
      if (this->Debug)
    	  fprintf(stderr,"\nUsing CUDA\n");
  vtkNewCUDAFFT* rfft2=vtkNewCUDAFFT::New();//implement CUDA's FFT/RFFT functionality
      rfft2->SetDebug(this->Debug);
  rfft2->SetForward(0);//set to RFFT
      rfft2->SetTimingMode(this->TimingMode);
      rfft2->SetDoComplex(this->DoComplexFFTs);
  rfft=rfft2;
    }
  else
    {
      if (this->Debug)
	fprintf(stderr,"\nUsing cpu\n");
      vtkImageFourierFilter* rfft2=vtkImageRFFT::New();
      rfft2->SetDimensionality(3);
      rfft=rfft2;
    }


  rfft->SetInput(math->GetOutput());
  math->Delete();

  if (this->Debug)
    fprintf(stderr, "\nComputing RFFT\n");

  rfft->Update();
  if (this->Debug){
    rfft->GetOutput()->GetDimensions(dim1);
    rfft->GetOutput()->GetOrigin(ori1);
    nc1=rfft->GetOutput()->GetNumberOfScalarComponents();
  }
  if (this->Debug)
    fprintf(stderr,"\t RFFT Done (%d x %d x %d, %d) ori=(%.1f,%.1f,%.1f)\n",
	    dim1[0],dim1[1],dim1[2],nc1,
	    ori1[0],ori1[1],ori1[2]);

#if defined(__unix) || defined(__linux)

  if (this->TimingMode){
    gettimeofday(&start, NULL);
    starttime=start.tv_sec+(start.tv_usec/1000000.0);
  }

#endif

  vtkImageCast* cast=vtkImageCast::New();
  cast->SetInput(rfft->GetOutput());
  cast->SetOutputScalarTypeToFloat();
  rfft->Delete();


#if defined(__unix) || defined(__linux)

  if (this->TimingMode){
    gettimeofday(&end, NULL);
    endtime=end.tv_sec+(end.tv_usec/1000000.0);
    fprintf(stderr, "\n  Casting took %.6f seconds.\n", endtime-starttime);
  }

#endif

  //  vtkImageData* tmp_img1=vtkImageData::New();
  vtkImageReslice* resl=vtkImageReslice::New();
  double ori[3];

#if defined(__unix) || defined(__linux)

  if (this->TimingMode){
    gettimeofday(&start, NULL);
    starttime=start.tv_sec+(start.tv_usec/1000000.0);
  }

#endif

  if (this->Mode==1 || dosquare == 1)
  {
      // Get the Magnitude
      // -----------------
      vtkImageMagnitude* magn=vtkImageMagnitude::New();
      magn->DebugOff();
      magn->SetInput(cast->GetOutput());
      magn->Update();
      resl->SetInput(magn->GetOutput());
      if (this->Debug)
	magn->GetOutput()->GetOrigin(ori);
      //  tmp_img1->ShallowCopy(magn->GetOutput());

      magn->Delete();
  }
  else
  {
      // Get the Real Part, convolution of two real images should be real
      // ----------------------------------------------------------------
      vtkImageExtractComponents* comp=vtkImageExtractComponents::New();
      comp->SetInput(cast->GetOutput());
      comp->SetComponents(0);
      comp->Update();
      resl->SetInput(comp->GetOutput());
      //    tmp_img1->ShallowCopy(comp->GetOutput());
      if (this->Debug)
    	  comp->GetOutput()->GetOrigin(ori);
      comp->Delete();

  }

#if defined(__unix) || defined(__linux)

  if (this->TimingMode){
    gettimeofday(&end, NULL);
    endtime=end.tv_sec+(end.tv_usec/1000000.0);
    fprintf(stderr, "\n  Getting magnitude/extracting components took %.6f seconds.\n", endtime-starttime);
    gettimeofday(&start, NULL);
    starttime=start.tv_sec+(start.tv_usec/1000000.0);
  }

#endif

  cast->Delete();
  resl->SetOutputExtent(0,origdim[0]-1,0,origdim[1]-1,0,origdim[2]-1);
  resl->SetOutputOrigin(origori);
  resl->SetOutputSpacing(img1->GetSpacing());
  resl->SetInterpolationMode(0);
  resl->SetBackgroundLevel(0.0);
  resl->Update();

  vtkImageData* out=vtkImageData::New();
  out->ShallowCopy(resl->GetOutput());
  resl->Delete();

#if defined(__unix) || defined(__linux)

  if (this->TimingMode){
    gettimeofday(&end, NULL);
    endtime=end.tv_sec+(end.tv_usec/1000000.0);
    fprintf(stderr, "\n  Reslicing and copying took %.6f seconds.\n", endtime-starttime);
  }

#endif

  if (this->Debug)
    {
      fprintf(stderr,"Reslice Input Origin = (%.3f,%.3f,%.3f)\n",ori[0],ori[1],ori[2]);
      fprintf(stderr,"Reslice Output Origin = (%.3f,%.3f,%.3f)\n",origori[0],origori[1],origori[2]);
      /*
      char line[100];
      sprintf(line,"zz_uncrop_%d.nii.gz",count);
      vtkpxUtil::SaveAnalyze(line,tmp_img1,9);
      fprintf(stderr,"uncropped image saved in %s\n",line);

      sprintf(line,"zz_crop_%d.nii.gz",count);
      vtkpxUtil::SaveAnalyze(line,out,9);
      fprintf(stderr,"uncropped image saved in %s\n",line);*/
    }

  //  tmp_img1->Delete();
  //  ++count;

  double range[2];
  if (this->Debug){
    out->GetDimensions(dim1);
    out->GetOrigin(ori1);
    nc1=out->GetNumberOfScalarComponents();

    out->GetPointData()->GetScalars()->GetRange(range);
  }
  if (this->Debug)
    fprintf(stderr,"\t Done F-Conv (%d x %d x %d, %d) ori=(%.1f,%.1f,%.1f) RANGE=%f:%f\n",
	    dim1[0],dim1[1],dim1[2],nc1,
	    ori1[0],ori1[1],ori1[2],range[0],range[1]);

  return out;
}

void vtkNewImageConvolution::normalizeFilters(vtkImageData* FILTERS){
  vtkDataArray* filters=FILTERS->GetPointData()->GetScalars();
  int frames=FILTERS->GetNumberOfScalarComponents();
  for (int i=0; i<frames; i++){
    int points=filters->GetNumberOfTuples();
    double sum=0.0;
    for (int j=0; j<points; j++){
      sum+=filters->GetComponent(j, i);
    }
    for (int j=0; j<points; j++){
      filters->SetComponent(j, i, (filters->GetComponent(j, i))/sum);
    }
  }
}

// ---------------------------------------------------------------------------------------
// --- should take more trivial ones (uniform sampling problem on spheres)
vtkImageData* vtkNewImageConvolution::qVesselFilter(double anglespacing,int radius,double sigma,double sigma_r)
{
  int rectmode=0;
  if (sigma<0.01 || sigma_r<0.01)
    {
      rectmode=1;
      // dummy values here
      sigma=1.0;
      sigma_r=1.0;
    }

  int dim=radius*2+1;
  double mid[3];
  for (int ia=0;ia<=2;ia++)
    mid[ia]=0.5*double(dim-1);


  int res=int(180.0/anglespacing+0.5);
  if (res<2)
    res=2;

  int numtheta=res*2;
  int numphi=res;
  int numfilters=(numtheta*numphi);

  vtkImageData* newimg=vtkImageData::New();
  newimg->SetScalarTypeToDouble();
  newimg->SetNumberOfScalarComponents(numfilters);
  newimg->SetDimensions(dim,dim,dim);
  newimg->AllocateScalars();


  vtkDataArray* data_arr=newimg->GetPointData()->GetScalars();
  int numtuples=data_arr->GetNumberOfTuples();

  // Fill Filter
  vtkDataArray *data=newimg->GetPointData()->GetScalars();
  for (int i=0;i<numfilters;i++)
    data->FillComponent(i,0.0);


  double pi=vtkMath::Pi();
  double kwd=anglespacing*pi/180.0;
  double halfkwd=kwd*0.5;

  double dradius=double(radius);
  double norm_r=1.0/(sqrt(2.0*pi)*sigma_r);
  double denom_r=2.0*sigma_r*sigma_r;

  //  sigma=sigma/(dradius*sin(kwd));

  sigma=sigma/(6.0*kwd);
  double norm  =1.0/(sqrt(2.0*pi)*sigma);
  double denom=2.0*sigma*sigma;

  int aindex = 0;
  for (int index=0;index<numphi;index++)
    {
      double sphi = index*kwd;
      double ephi = (index+1)*kwd;
      double mphi = sphi + halfkwd;
      for (int indt=0; indt<numtheta; indt++)
	{
	  double stht = indt*kwd;
	  double etht = (indt+1)*kwd;
	  double mtht = stht + halfkwd;


	  double sumframe=0.0;
	  for (int k=0;k<dim;k++)
	    {
	      double p[3];
	      p[2]=double(k)-mid[2];
	      for (int j=0;j<dim;j++)
		{
		  p[1]=double(j)-mid[1];
		  for (int i=0;i<dim;i++)
		    {
		      p[0]=double(i)-mid[0];

		      double r=vtkMath::Norm(p); //+0.0001;
		      double g=1.0;

		      // another way:
		      double cs;
		      if (r>0.0)
			cs = acos(p[2]/r);
		      else
			cs = 0.0;

		      double xcs;
		      double xr = sqrt(p[0]*p[0]+p[1]*p[1]);
		      if (xr > 0.0)
			{
			  xcs = acos(p[0]/xr);
			  if (p[1]<0.0)
			    xcs = 2*pi - xcs;
			}
		      else
			xcs = 0.0;


		      double normalized_r=r/dradius;

		      if (rectmode==0)
			g=norm_r*exp(-pow(normalized_r,2.0)/denom_r);
		      else if (normalized_r>1.0)
			g=0.0;

		      if (rectmode==0)
			{
			  double normalized_d = fabs(cs-mphi);
			  double normalized_xd = fabs(xcs-mtht);
			  if (normalized_xd > fabs(xcs-mtht+2*pi))
			    normalized_xd = fabs(xcs-mtht+2*pi);
			  if (normalized_xd > fabs(xcs-mtht-2*pi))
			    normalized_xd = fabs(xcs-mtht-2*pi);

			  g*=norm*exp(-(pow(normalized_d,2.0)+pow(normalized_xd,2.0))/denom);
			}
		      else
			{
			  if ((cs >= ephi) || (cs < sphi))
		    g=0.0;
			  else
			    {
			      if ((xcs >= etht) || (xcs < stht))
				g = 0.0;
			    }
			}

		      sumframe+=g;
		      newimg->SetScalarComponentFromDouble(i,j,k,aindex,g);
		    }
		}
	    }

	  for (int i=0;i<numtuples;i++)
	    data_arr->SetComponent(i,aindex,data_arr->GetComponent(i,aindex)/sumframe);
	  ++aindex;
	}
    }

  return newimg;
}
