/*
 * vtkjoColoredSphereSource.cpp
 *
 *  Created on: Oct 28, 2008
 *      Author: John Onofrey
 */

#include "vtkjoColoredSphereSource.h"

#include "vtkCellArray.h"
#include "vtkDoubleArray.h"
#include "vtkImageData.h"
#include "vtkImageReslice.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkjoSphereSource.h"
#include "vtkObjectFactory.h"
#include "vtkPointData.h"
#include "vtkPolyData.h"
#include "vtkPolyDataToImageStencil.h"
#include "vtkShortArray.h"
#include "vtkStreamingDemandDrivenPipeline.h"
#include <assert.h>

vtkjoColoredSphereSource* vtkjoColoredSphereSource::New()
{
    // First try to create the object from the vtkObjectFactory
    vtkObject* ret = vtkObjectFactory::CreateInstance("vtkjoColoredSphereSource");
    if (ret)
    {
        return (vtkjoColoredSphereSource*) ret;
    }
    // If the factory was unable to create the object, then create it here.
    return new vtkjoColoredSphereSource;
}

//---------------------------------------------------------------------------

vtkjoColoredSphereSource::vtkjoColoredSphereSource()
{
    this->SetNumberOfInputPorts(0);

    this->UseNewWedges=0;

    this->Radius = 32.0;
    this->SubdivisionLevels = 0;

    this->Center[0] = 32.0;
    this->Center[1] = 32.0;
    this->Center[2] = 32.0;

    this->WholeExtent[0] = 0;  this->WholeExtent[1] = 64;
    this->WholeExtent[2] = 0;  this->WholeExtent[3] = 64;
    this->WholeExtent[4] = 0;  this->WholeExtent[5] = 64;

    this->UseSingleFrameOff();
    this->ColorValue = 100;
}

//---------------------------------------------------------------------------

void vtkjoColoredSphereSource::PrintSelf(ostream& os, vtkIndent indent)
{
    this->Superclass::PrintSelf(os, indent);

    os << indent << "Radius: " << this->Radius << "\n";
    os << indent << "SubdivisionLevels: " << this->SubdivisionLevels << "\n";
    os << indent << "ColorValue: " << this->ColorValue << "\n";
    os << indent << "UseSingleFrame: " << this->UseSingleFrame << "\n";
}

//---------------------------------------------------------------------------

void vtkjoColoredSphereSource::SetRadius(double r)
{
    if (r < 0.0) {
        vtkErrorMacro("SetRadius: Input radius is negative - multiplying by -1");
        r *= -1.0;
    }

    if (this->Radius != r)
    {
        this->Radius = r;

        // Readjust the Center and WholeExtent
        int rMax = int(ceil(r));

        this->Center[0] = double(rMax);
        this->Center[1] = double(rMax);
        this->Center[2] = double(rMax);

        this->WholeExtent[0] = 0;  this->WholeExtent[1] = 2*rMax;
        this->WholeExtent[2] = 0;  this->WholeExtent[3] = 2*rMax;
        this->WholeExtent[4] = 0;  this->WholeExtent[5] = 2*rMax;

        this->Modified();
    }
}

//---------------------------------------------------------------------------

int vtkjoColoredSphereSource::RequestInformation(vtkInformation* vtkNotUsed(request),
        vtkInformationVector** vtkNotUsed(inputVector),
        vtkInformationVector* outputVector)
{
    vtkInformation* outInfo = outputVector->GetInformationObject(0);
    outInfo->Set(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT(),
            this->WholeExtent, 6);


    if (!this->UseSingleFrame)
    {
        // Calculate the number of faces of the sphere surface
        int numFaces = 20*int(pow(float(4), this->SubdivisionLevels));
        vtkDataObject::SetPointDataActiveScalarInfo(outInfo, VTK_SHORT, numFaces);
    }
    else
        vtkDataObject::SetPointDataActiveScalarInfo(outInfo, VTK_SHORT, 1);

    return 1;
}

//---------------------------------------------------------------------------

int vtkjoColoredSphereSource::RequestData(vtkInformation* vtkNotUsed(request),
        vtkInformationVector** vtkNotUsed(inputVector),
        vtkInformationVector* outputVector)
{
    vtkInformation *outInfo = outputVector->GetInformationObject(0);
    vtkImageData *output = vtkImageData::SafeDownCast(
            outInfo->Get(vtkDataObject::DATA_OBJECT()));
    vtkImageData *data = this->AllocateOutputData(output);

    if (data->GetScalarType() != VTK_SHORT)
    {
        vtkErrorMacro("Execute: This source only outputs shorts");
    }
    vtkShortArray* outData = static_cast<vtkShortArray*>(data->GetPointData()->GetScalars());
    vtkIdType numPoints = data->GetNumberOfPoints();
    int numComps = outData->GetNumberOfComponents();
    // Initialize all components to zero
    for (int i=0; i<numComps; i++)
        outData->FillComponent(i, 0.0);

    short* outData_ptr = outData->GetPointer(0);

    // Create an image of all 1's to create the stencil intersection with
    // the geometry
    vtkImageData* onesImg=vtkImageData::New();
    onesImg->CopyStructure(data);
    onesImg->AllocateScalars();
    onesImg->GetPointData()->GetScalars()->FillComponent(0,1.0);

    // Create the tessellated sphere
    vtkjoSphereSource* sphereSrc = vtkjoSphereSource::New();
    sphereSrc->SetRadius(this->Radius);
    sphereSrc->SetCenter(this->Center[0], this->Center[1], this->Center[2]);
    sphereSrc->IcosahedralTessellationOn();
    sphereSrc->SetSubdivisionLevels(this->SubdivisionLevels);
    sphereSrc->Update();
    vtkPolyData* sphere = sphereSrc->GetOutput();

    // Get each face of the surface
    vtkIdType numFaces = sphere->GetNumberOfCells();
    vtkCellArray* faces = sphere->GetPolys();
    faces->InitTraversal();

    // Get the surface points
    vtkPoints* points = sphere->GetPoints();

    vtkPolyDataToImageStencil* stencil = vtkPolyDataToImageStencil::New();
    stencil->SetInformationInput(onesImg);

    vtkImageReslice* reslice = vtkImageReslice::New();
    reslice->SetInput(onesImg);

    vtkIdType nPts;
    vtkIdType* pts = new vtkIdType[3];
    double p0[3], p1[3], p2[3];
    double q0[3], q1[3], q2[3];
    double opp0[3], opp1[3], opp2[3];
    for (int i=0; i<numFaces; i++)
    {
        faces->GetNextCell(nPts, pts);

        points->GetPoint(pts[0], p0);
        points->GetPoint(pts[1], p1);
        points->GetPoint(pts[2], p2);

	if (this->UseNewWedges){
	  double centersum=this->Center[0]+this->Center[1]+this->Center[2];
	  double centroid[3];
	  for (int x=0; x<3; x++){
	    //reflect p_i through this->Center
	    opp0[x]=2*this->Center[x]-p0[x];
	    opp1[x]=2*this->Center[x]-p1[x];
	    opp2[x]=2*this->Center[x]-p2[x];
	    //reflect opp_i through centroid of triangle T=(opp_0, opp_1, opp_2) [equivalent to rotation by 180]
	    centroid[x]=(opp0[x]+opp1[x]+opp2[x])/3.0;
	    q0[x]=2*centroid[x]-opp0[x];
	    q1[x]=2*centroid[x]-opp1[x];
	    q2[x]=2*centroid[x]-opp2[x];
	  }
	}

        // Setup the wedge points
        vtkPoints* wedgePts = vtkPoints::New();

	if (this->UseNewWedges){
	  wedgePts->Allocate(6, 0);
	  wedgePts->InsertPoint(0, p0);
	  wedgePts->InsertPoint(1, p1);
	  wedgePts->InsertPoint(2, p2);
	  wedgePts->InsertPoint(3, q0);
	  wedgePts->InsertPoint(4, q1);
	  wedgePts->InsertPoint(5, q2);
	}
	else {
	  wedgePts->Allocate(4, 0);
	  wedgePts->InsertPoint(0, p0);
	  wedgePts->InsertPoint(1, p1);
	  wedgePts->InsertPoint(2, p2);
	  wedgePts->InsertPoint(3, this->Center);
	}

	// Setup the wedge cells
        vtkCellArray* wedgePolys = vtkCellArray::New();

	if (this->UseNewWedges){
	  wedgePolys->Allocate(wedgePolys->EstimateSize(5, 4));
	  vtkIdType* rects=new vtkIdType[4];
	  rects[0]=0;
	  rects[1]=1;
	  rects[2]=4;
	  rects[3]=3;
	  wedgePolys->InsertNextCell(4, rects);
	  rects[0]=0;
	  rects[1]=2;
	  rects[2]=5;
	  rects[3]=3;
	  wedgePolys->InsertNextCell(4, rects);
	  rects[0]=1;
	  rects[1]=2;
	  rects[2]=5;
	  rects[3]=4;
	  wedgePolys->InsertNextCell(4, rects);
	  vtkIdType* triangles=new vtkIdType[3];
	  triangles[0]=0;
	  triangles[1]=1;
	  triangles[2]=2;
	  wedgePolys->InsertNextCell(3, triangles);
	  triangles[0]=3;
	  triangles[1]=4;
	  triangles[2]=5;
	  wedgePolys->InsertNextCell(3, triangles);
	  wedgePolys->Squeeze();
	  delete[] rects;
	  delete[] triangles;
	}
	else {
	  wedgePolys->Allocate(wedgePolys->EstimateSize(4, 3));
	  vtkIdType* ids = new vtkIdType[3];
	  ids[0] = 0;
	  ids[1] = 1;
	  ids[2] = 2;
	  wedgePolys->InsertNextCell(nPts, ids);
	  ids[0] = 3;
	  ids[1] = 1;
	  ids[2] = 0;
	  wedgePolys->InsertNextCell(nPts, ids);
	  ids[0] = 3;
	  ids[1] = 0;
	  ids[2] = 2;
	  wedgePolys->InsertNextCell(nPts, ids);
	  ids[0] = 3;
	  ids[1] = 2;
	  ids[2] = 1;
	  wedgePolys->InsertNextCell(nPts, ids);
	  delete[] ids;
	}

        // Create the wedge PolyData out of the points and cells
        vtkPolyData* wedge = vtkPolyData::New();
        wedge->SetPoints(wedgePts);
        wedge->SetPolys(wedgePolys);
        wedgePts->Delete();
        wedgePolys->Delete();

        // Create a stencil from the wedge PolyData and then reslice to
        // get the wedge as image data
        stencil->SetInput(wedge);
        stencil->Update();

        reslice->SetStencil(stencil->GetOutput());
        reslice->SetBackgroundColor(0.0, 0.0, 0.0, 1.0);
        reslice->Update();

        vtkImageData* wedgeImg = reslice->GetOutput();
        vtkDoubleArray* wedgeData = (vtkDoubleArray*) wedgeImg->GetPointData()->GetScalars();
        double* wedgeData_ptr = wedgeData->GetPointer(0);

        // Color the output according the wedge image
        if (!this->UseSingleFrame)
        {
            for (int j=0; j<numPoints; j++)
            {
                if (wedgeData_ptr[j] > 0)
                    outData_ptr[i+j*numComps] = ColorValue;
            }

        }
        else
        {
            int color = i+1;
            for (int j=0; j<numPoints; j++)
            {
                if (wedgeData_ptr[j] > 0 && !(outData_ptr[j] > 0))
                    outData_ptr[j] = color;
            }
        }
    }

    stencil->Delete();
    reslice->Delete();
    onesImg->Delete();
    sphereSrc->Delete();

    return 1;
}
