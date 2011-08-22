#ifndef __vtkjoSphereSource_h
#define __vtkjoSphereSource_h

#include "vtkPolyDataAlgorithm.h"

#define VTK_MAX_SPHERE_RESOLUTION 1024
#define VTK_MAX_SPHERE_SUBDIVISIONS 8

class vtkjoSphereSource : public vtkPolyDataAlgorithm
{
public:
  vtkTypeRevisionMacro(vtkjoSphereSource,vtkPolyDataAlgorithm);
  void PrintSelf(ostream& os, vtkIndent indent);

  // Description:
  // Construct sphere with radius=0.5 and default resolution 8 in both Phi
  // and Theta directions. Theta ranges from (0,360) and phi (0,180) degrees.
  static vtkjoSphereSource *New();

  // Description:
  // Set radius of sphere. Default is .5.
  vtkSetClampMacro(Radius,double,0.0,VTK_DOUBLE_MAX);
  vtkGetMacro(Radius,double);

  // Description:
  // Set the center of the sphere. Default is 0,0,0.
  vtkSetVector3Macro(Center,double);
  vtkGetVectorMacro(Center,double,3);

  // Description:
  // Set the number of points in the longitude direction (ranging from
  // StartTheta to EndTheta).
  vtkSetClampMacro(ThetaResolution,int,3,VTK_MAX_SPHERE_RESOLUTION);
  vtkGetMacro(ThetaResolution,int);

  // Description:
  // Set the number of points in the latitude direction (ranging
  // from StartPhi to EndPhi).
  vtkSetClampMacro(PhiResolution,int,3,VTK_MAX_SPHERE_RESOLUTION);
  vtkGetMacro(PhiResolution,int);

  // Description:
  // Set the starting longitude angle. By default StartTheta=0 degrees.
  vtkSetClampMacro(StartTheta,double,0.0,360.0);
  vtkGetMacro(StartTheta,double);

  // Description:
  // Set the ending longitude angle. By default EndTheta=360 degrees.
  vtkSetClampMacro(EndTheta,double,0.0,360.0);
  vtkGetMacro(EndTheta,double);

  // Description:
  // Set the starting latitude angle (0 is at north pole). By default
  // StartPhi=0 degrees.
  vtkSetClampMacro(StartPhi,double,0.0,360.0);
  vtkGetMacro(StartPhi,double);

  // Description:
  // Set the ending latitude angle. By default EndPhi=180 degrees.
  vtkSetClampMacro(EndPhi,double,0.0,360.0);
  vtkGetMacro(EndPhi,double);

  // Description:
  // Cause the sphere to be tessellated with edges along the latitude
  // and longitude lines. If off, triangles are generated at non-polar
  // regions, which results in edges that are not parallel to latitude and
  // longitude lines. If on, quadrilaterals are generated everywhere
  // except at the poles. This can be useful for generating a wireframe
  // sphere with natural latitude and longitude lines.
  vtkSetMacro(LatLongTessellation,int);
  vtkGetMacro(LatLongTessellation,int);
  vtkBooleanMacro(LatLongTessellation,int);

  // Description:
  // Start with an icosahedron rather than use longitude and latitude
  // style for constructing the sphere.  The vertices are more uniformly
  // distributed throughout the sphere surface.
  vtkSetMacro(IcosahedralTessellation, int);
  vtkGetMacro(IcosahedralTessellation, int);
  vtkBooleanMacro(IcosahedralTessellation, int);

  // Description:
  // Set the number of subdivisions to perform on the icosahedron when creating
  // the sphere.  0 subdivions will return the initial icosahedron.
  vtkSetClampMacro(SubdivisionLevels, int, 0, VTK_MAX_SPHERE_SUBDIVISIONS);
  vtkGetMacro(SubdivisionLevels, int);

protected:
  vtkjoSphereSource(int res=8);
  ~vtkjoSphereSource() {}

  int RequestData(vtkInformation *, vtkInformationVector **, vtkInformationVector *);
  int RequestInformation(vtkInformation *, vtkInformationVector **, vtkInformationVector *);

  double Radius;
  double Center[3];
  int ThetaResolution;
  int PhiResolution;
  double StartTheta;
  double EndTheta;
  double StartPhi;
  double EndPhi;
  int LatLongTessellation;
  int IcosahedralTessellation;
  int SubdivisionLevels;

  int PerformLatLongTessellation(vtkInformationVector *);
  int PerformIcosahedralTessellation(vtkInformationVector *);

private:
  vtkjoSphereSource(const vtkjoSphereSource&);  // Not implemented.
  void operator=(const vtkjoSphereSource&);  // Not implemented.
};

#endif
