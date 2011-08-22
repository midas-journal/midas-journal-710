#include "vtkjoSphereSource.h"

#include "vtkCellArray.h"
#include "vtkEdgeTable.h"
#include "vtkFloatArray.h"
#include "vtkDoubleArray.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkMath.h"
#include "vtkObjectFactory.h"
#include "vtkPlatonicSolidSource.h"
#include "vtkPointData.h"
#include "vtkPoints.h"
#include "vtkPolyData.h"
#include "vtkStreamingDemandDrivenPipeline.h"

#include <math.h>

vtkCxxRevisionMacro(vtkjoSphereSource, "$Revision: 1.0 $");
vtkStandardNewMacro(vtkjoSphereSource);

//----------------------------------------------------------------------------
// Construct sphere with radius=0.5 and default resolution 8 in both Phi
// and Theta directions. Theta ranges from (0,360) and phi (0,180) degrees.
vtkjoSphereSource::vtkjoSphereSource(int res)
{
  res = res < 4 ? 4 : res;
  this->Radius = 0.5;
  this->Center[0] = 0.0;
  this->Center[1] = 0.0;
  this->Center[2] = 0.0;

  this->ThetaResolution = res;
  this->PhiResolution = res;
  this->StartTheta = 0.0;
  this->EndTheta = 360.0;
  this->StartPhi = 0.0;
  this->EndPhi = 180.0;
  this->LatLongTessellation = 0;
  this->IcosahedralTessellation = 0;
  this->SubdivisionLevels = 3;

  this->SetNumberOfInputPorts(0);
}

//----------------------------------------------------------------------------
int vtkjoSphereSource::RequestData(
  vtkInformation *vtkNotUsed(request),
  vtkInformationVector **vtkNotUsed(inputVector),
  vtkInformationVector *outputVector)
{
  if (this->IcosahedralTessellation < 1) {
	  return PerformLatLongTessellation(outputVector);
  } else {
	  return PerformIcosahedralTessellation(outputVector);
  }
}

//----------------------------------------------------------------------------
int vtkjoSphereSource::PerformLatLongTessellation(vtkInformationVector *outputVector)
{
	  // get the info object
	  vtkInformation *outInfo = outputVector->GetInformationObject(0);

	  // get the output
	  vtkPolyData *output = vtkPolyData::SafeDownCast(
	    outInfo->Get(vtkDataObject::DATA_OBJECT()));

	  int i, j;
	  int jStart, jEnd, numOffset;
	  int numPts, numPolys;
	  vtkPoints *newPoints;
	  vtkFloatArray *newNormals;
	  vtkCellArray *newPolys;
	  double x[3], n[3], deltaPhi, deltaTheta, phi, theta, radius, norm;
	  double startTheta, endTheta, startPhi, endPhi;
	  int base, numPoles=0, thetaResolution, phiResolution;
	  vtkIdType pts[4];
	  int piece =
	    outInfo->Get(vtkStreamingDemandDrivenPipeline::UPDATE_PIECE_NUMBER());
	  int numPieces =
	    outInfo->Get(vtkStreamingDemandDrivenPipeline::UPDATE_NUMBER_OF_PIECES());

	  if (numPieces > this->ThetaResolution)
	    {
	    numPieces = this->ThetaResolution;
	    }
	  if (piece >= numPieces)
	    {
	    // Although the super class should take care of this,
	    // it cannot hurt to check here.
	    return 1;
	    }

	  // I want to modify the ivars resoultion start theta and end theta,
	  // so I will make local copies of them.  THese might be able to be merged
	  // with the other copies of them, ...
	  int localThetaResolution = this->ThetaResolution;
	  double localStartTheta = this->StartTheta;
	  double localEndTheta = this->EndTheta;

	  while (localEndTheta < localStartTheta)
	    {
	    localEndTheta += 360.0;
	    }
	  deltaTheta = (localEndTheta - localStartTheta) / localThetaResolution;

	  // Change the ivars based on pieces.
	  int start, end;
	  start = piece * localThetaResolution / numPieces;
	  end = (piece+1) * localThetaResolution / numPieces;
	  localEndTheta = localStartTheta + (double)(end) * deltaTheta;
	  localStartTheta = localStartTheta + (double)(start) * deltaTheta;
	  localThetaResolution = end - start;

	  // Set things up; allocate memory
	  //
	  vtkDebugMacro("SphereSource Executing piece index " << piece
	                << " of " << numPieces << " pieces.");

	  numPts = this->PhiResolution * localThetaResolution + 2;
	  // creating triangles
	  numPolys = this->PhiResolution * 2 * localThetaResolution;

	  newPoints = vtkPoints::New();
	  newPoints->Allocate(numPts);
	  newNormals = vtkFloatArray::New();
	  newNormals->SetNumberOfComponents(3);
	  newNormals->Allocate(3*numPts);
	  newNormals->SetName("Normals");

	  newPolys = vtkCellArray::New();
	  newPolys->Allocate(newPolys->EstimateSize(numPolys, 3));

	  // Create sphere
	  //
	  // Create north pole if needed
	  if ( this->StartPhi <= 0.0 )
	    {
	    x[0] = this->Center[0];
	    x[1] = this->Center[1];
	    x[2] = this->Center[2] + this->Radius;
	    newPoints->InsertPoint(numPoles,x);

	    x[0] = x[1] = 0.0; x[2] = 1.0;
	    newNormals->InsertTuple(numPoles,x);
	    numPoles++;
	    }

	  // Create south pole if needed
	  if ( this->EndPhi >= 180.0 )
	    {
	    x[0] = this->Center[0];
	    x[1] = this->Center[1];
	    x[2] = this->Center[2] - this->Radius;
	    newPoints->InsertPoint(numPoles,x);

	    x[0] = x[1] = 0.0; x[2] = -1.0;
	    newNormals->InsertTuple(numPoles,x);
	    numPoles++;
	    }

	  // Check data, determine increments, and convert to radians
	  startTheta = (localStartTheta < localEndTheta ? localStartTheta : localEndTheta);
	  startTheta *= vtkMath::Pi() / 180.0;
	  endTheta = (localEndTheta > localStartTheta ? localEndTheta : localStartTheta);
	  endTheta *= vtkMath::Pi() / 180.0;

	  startPhi = (this->StartPhi < this->EndPhi ? this->StartPhi : this->EndPhi);
	  startPhi *= vtkMath::Pi() / 180.0;
	  endPhi = (this->EndPhi > this->StartPhi ? this->EndPhi : this->StartPhi);
	  endPhi *= vtkMath::Pi() / 180.0;

	  phiResolution = this->PhiResolution - numPoles;
	  deltaPhi = (endPhi - startPhi) / (this->PhiResolution - 1);
	  thetaResolution = localThetaResolution;
	  if (fabs(localStartTheta - localEndTheta) < 360.0)
	    {
	    ++localThetaResolution;
	    }
	  deltaTheta = (endTheta - startTheta) / thetaResolution;

	  jStart = (this->StartPhi <= 0.0 ? 1 : 0);
	  jEnd = (this->EndPhi >= 180.0 ? this->PhiResolution - 1
	        : this->PhiResolution);

	  this->UpdateProgress(0.1);

	  // Create intermediate points
	  for (i=0; i < localThetaResolution; i++)
	    {
	    theta = localStartTheta * vtkMath::Pi() / 180.0 + i*deltaTheta;

	    for (j=jStart; j<jEnd; j++)
	      {
	      phi = startPhi + j*deltaPhi;
	      radius = this->Radius * sin((double)phi);
	      n[0] = radius * cos((double)theta);
	      n[1] = radius * sin((double)theta);
	      n[2] = this->Radius * cos((double)phi);
	      x[0] = n[0] + this->Center[0];
	      x[1] = n[1] + this->Center[1];
	      x[2] = n[2] + this->Center[2];
	      newPoints->InsertNextPoint(x);

	      if ( (norm = vtkMath::Norm(n)) == 0.0 )
	        {
	        norm = 1.0;
	        }
	      n[0] /= norm; n[1] /= norm; n[2] /= norm;
	      newNormals->InsertNextTuple(n);
	      }
	    this->UpdateProgress (0.10 + 0.50*i/static_cast<float>(localThetaResolution));
	    }

	  // Generate mesh connectivity
	  base = phiResolution * localThetaResolution;

	  if (fabs(localStartTheta - localEndTheta) < 360.0)
	    {
	    --localThetaResolution;
	    }

	  if ( this->StartPhi <= 0.0 )  // around north pole
	    {
	    for (i=0; i < localThetaResolution; i++)
	      {
	      pts[0] = phiResolution*i + numPoles;
	      pts[1] = (phiResolution*(i+1) % base) + numPoles;
	      pts[2] = 0;
	      newPolys->InsertNextCell(3, pts);
	      }
	    }

	  if ( this->EndPhi >= 180.0 ) // around south pole
	    {
	    numOffset = phiResolution - 1 + numPoles;

	    for (i=0; i < localThetaResolution; i++)
	      {
	      pts[0] = phiResolution*i + numOffset;
	      pts[2] = ((phiResolution*(i+1)) % base) + numOffset;
	      pts[1] = numPoles - 1;
	      newPolys->InsertNextCell(3, pts);
	      }
	    }
	  this->UpdateProgress (0.70);

	  // bands in-between poles
	  for (i=0; i < localThetaResolution; i++)
	    {
	    for (j=0; j < (phiResolution-1); j++)
	      {
	      pts[0] = phiResolution*i + j + numPoles;
	      pts[1] = pts[0] + 1;
	      pts[2] = ((phiResolution*(i+1)+j) % base) + numPoles + 1;
	      if ( !this->LatLongTessellation )
	        {
	        newPolys->InsertNextCell(3, pts);
	        pts[1] = pts[2];
	        pts[2] = pts[1] - 1;
	        newPolys->InsertNextCell(3, pts);
	        }
	      else
	        {
	        pts[3] = pts[2] - 1;
	        newPolys->InsertNextCell(4, pts);
	        }
	      }
	    this->UpdateProgress (0.70 + 0.30*i/static_cast<double>(localThetaResolution));
	    }

	  // Update ourselves and release memeory
	  //
	  newPoints->Squeeze();
	  output->SetPoints(newPoints);
	  newPoints->Delete();

	  newNormals->Squeeze();
	  output->GetPointData()->SetNormals(newNormals);
	  newNormals->Delete();

	  newPolys->Squeeze();
	  output->SetPolys(newPolys);
	  newPolys->Delete();

	return 1;
}



//----------------------------------------------------------------------------
int vtkjoSphereSource::PerformIcosahedralTessellation(vtkInformationVector *outputVector)
{
	// get the info object
	vtkInformation *outInfo = outputVector->GetInformationObject(0);

	// get the output
	vtkPolyData *output = vtkPolyData::SafeDownCast(
			outInfo->Get(vtkDataObject::DATA_OBJECT()));

	vtkDebugMacro(<<"Creating the initial icosahedron");

	vtkPlatonicSolidSource* icosaSrc = vtkPlatonicSolidSource::New();
	icosaSrc->SetSolidTypeToIcosahedron();
	icosaSrc->Update();

	vtkPolyData* original = vtkPolyData::New();
	original->ShallowCopy(icosaSrc->GetOutput());
	icosaSrc->Delete();

	vtkPoints* points = original->GetPoints();
	vtkIdType numPts = points->GetNumberOfPoints();
	vtkIdType numEdges = 30;	// Known for an icosahedron

	vtkIdType numCells = original->GetNumberOfCells();
	vtkCellArray* polys = original->GetPolys();

	this->UpdateProgress(0.1);

	for (int level=0; level<this->SubdivisionLevels; level++) {
		vtkDebugMacro(<<"numPoints="<<numPts<<" numCells="<<numCells<<" numEdges="<<numEdges);

		// The number of points in the new sphere will be numPts + numEdges
		vtkPoints* newPoints = vtkPoints::New();
		newPoints->Allocate(numPts + numEdges, 0);
		newPoints->ShallowCopy(points);

		// Create a table to maintain a list of all edges in the surface
		vtkEdgeTable* edgeTable = vtkEdgeTable::New();
        edgeTable->InitEdgeInsertion(numPts, 1);

        vtkIdType nPts;
		vtkIdType* pts = new vtkIdType[3];

		vtkIdType edgeId;
		double mid[3];
		double p1[3], p2[3];
		vtkIdType nextIdx;
		double norm;

		// Traverse the cells and record edges while inserting midpoints
		// into the midpoint array if the edge does not exist in the edge
		// table
		polys->InitTraversal();
		for (int i=0; i<numCells; i++) {
			polys->GetNextCell(nPts, pts);
			for (int idx=0; idx<nPts; idx++) {
				nextIdx = (idx+1)%nPts;
				vtkIdType id = edgeTable->IsEdge(pts[idx], pts[nextIdx]);
				if (id == -1) {
					points->GetPoint(pts[idx], p1);
					points->GetPoint(pts[nextIdx], p2);
					mid[0] = (p1[0] + p2[0])/2.0;
					mid[1] = (p1[1] + p2[1])/2.0;
					mid[2] = (p1[2] + p2[2])/2.0;

					// Normalize the this midpoint vector to the unit sphere
					// by dividing by the norm
					if ((norm = vtkMath::Norm(mid)) == 0) {
						norm = 1.0;
					}
					mid[0] /= norm;
					mid[1] /= norm;
					mid[2] /= norm;

					edgeId = newPoints->InsertNextPoint(mid);

					// The edge is not in the table, so insert it!
					edgeTable->InsertEdge(pts[idx], pts[nextIdx], edgeId);
				}
			}
		}

		// Now, create the new cells using the new points

		// Create a new cell array to store polygon information.
		// The new surface will subdivide each cell into 4 new cells, with
		// each cell being defined by 3 points.
		vtkCellArray* newPolys = vtkCellArray::New();
		newPolys->Allocate(newPolys->EstimateSize(4*numCells, 3));

		vtkIdType edgeIdA;
		vtkIdType edgeIdB;
		vtkIdType edgeIdC;
		vtkIdType* ids = new vtkIdType[3];

		polys->InitTraversal();
		for (int i=0; i<numCells; i++) {
			polys->GetNextCell(nPts, pts);
			// Assume pts contains 3 ids
			edgeIdA = edgeTable->IsEdge(pts[0], pts[1]);
			edgeIdB = edgeTable->IsEdge(pts[1], pts[2]);
			edgeIdC = edgeTable->IsEdge(pts[2], pts[0]);

			ids[0] = pts[0];
			ids[1] = edgeIdA;
			ids[2] = edgeIdC;
			newPolys->InsertNextCell(nPts, ids);
			ids[0] = edgeIdA;
			ids[1] = pts[1];
			ids[2] = edgeIdB;
			newPolys->InsertNextCell(nPts, ids);
			ids[0] = edgeIdC;
			ids[1] = edgeIdB;
			ids[2] = pts[2];
			newPolys->InsertNextCell(nPts, ids);
			ids[0] = edgeIdA;
			ids[1] = edgeIdB;
			ids[2] = edgeIdC;
			newPolys->InsertNextCell(nPts, ids);
		}

		// Update the point, cell, and edge counts
		numPts = points->GetNumberOfPoints();	  // numPts = numPts + numEdges
		numEdges = 2*numEdges + 3*numCells;		  // numEdges = 2*numEdges + 3*numCells
		numCells = newPolys->GetNumberOfCells();  // numCells = 4*numCells

		this->UpdateProgress (0.10 + 0.80*level/static_cast<double>(this->SubdivisionLevels));

		// Cleanup at the end of each iteration.
		edgeTable->Delete();

		delete [] ids;

		points->ShallowCopy(newPoints);
		newPoints->Delete();
		polys->DeepCopy(newPolys);
		newPolys->Delete();
	}

	vtkDebugMacro(<<"numPoints="<<numPts<<" numCells="<<numCells<<" numEdges="<<numEdges);

	// Add point normals
	vtkDoubleArray* newNormals = vtkDoubleArray::New();
	newNormals->SetNumberOfComponents(3);
	newNormals->Allocate(3*numPts);
	newNormals->SetName("Normals");
	double p[3], n[3];
	double norm;

	for (int i=0; i<numPts; i++) {
		points->GetPoint(i, p);
		n[0] = p[0];
		n[1] = p[1];
		n[2] = p[2];
		if ((norm = vtkMath::Norm(n)) == 0.0) {
			norm = 1.0;
		}
		n[0] /= -1.0*norm;
		n[1] /= -1.0*norm;
		n[2] /= -1.0*norm;
		newNormals->InsertNextTuple(n);

		// While we're here, scale to the set radius and translate
		// to the specified center
		p[0] = this->Radius*p[0] + this->Center[0];
		p[1] = this->Radius*p[1] + this->Center[1];
		p[2] = this->Radius*p[2] + this->Center[2];
		points->SetPoint(i, p);
	}

	points->Squeeze();
	output->SetPoints(points);
	points->Delete();

	newNormals->Squeeze();
	output->GetPointData()->SetNormals(newNormals);

	newNormals->Delete();

	polys->Squeeze();
	output->SetPolys(polys);
	polys->Delete();

	this->UpdateProgress(1.0);

	return 1;
}


//----------------------------------------------------------------------------
void vtkjoSphereSource::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);

  os << indent << "Theta Resolution: " << this->ThetaResolution << "\n";
  os << indent << "Phi Resolution: " << this->PhiResolution << "\n";
  os << indent << "Theta Start: " << this->StartTheta << "\n";
  os << indent << "Phi Start: " << this->StartPhi << "\n";
  os << indent << "Theta End: " << this->EndTheta << "\n";
  os << indent << "Phi End: " << this->EndPhi << "\n";
  os << indent << "Radius: " << this->Radius << "\n";
  os << indent << "Center: (" << this->Center[0] << ", "
     << this->Center[1] << ", " << this->Center[2] << ")\n";
  os << indent
     << "LatLong Tessellation: " << this->LatLongTessellation << "\n";
  os << indent
	 << "Icosahedral Tessellation: " << this->IcosahedralTessellation << "\n";
  os << indent << "SubdivisionLevels: " << this->SubdivisionLevels << "\n";
}

//----------------------------------------------------------------------------
int vtkjoSphereSource::RequestInformation(
  vtkInformation *vtkNotUsed(request),
  vtkInformationVector **vtkNotUsed(inputVector),
  vtkInformationVector *outputVector)
{
  // get the info object
  vtkInformation *outInfo = outputVector->GetInformationObject(0);

  outInfo->Set(vtkStreamingDemandDrivenPipeline::MAXIMUM_NUMBER_OF_PIECES(),
               -1);

  outInfo->Set(vtkStreamingDemandDrivenPipeline::WHOLE_BOUNDING_BOX(),
               this->Center[0] - this->Radius,
               this->Center[0] + this->Radius,
               this->Center[1] - this->Radius,
               this->Center[1] + this->Radius,
               this->Center[2] - this->Radius,
               this->Center[2] + this->Radius);

  return 1;
}
