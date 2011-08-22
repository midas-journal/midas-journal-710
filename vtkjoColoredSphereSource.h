
#ifndef VTKJOCOLOREDSPHERESOURCE_H_
#define VTKJOCOLOREDSPHERESOURCE_H_

#include "vtkImageAlgorithm.h"

class vtkjoColoredSphereSource : public vtkImageAlgorithm
{
public:
    static vtkjoColoredSphereSource *New();
    vtkTypeMacro(vtkjoColoredSphereSource, vtkImageAlgorithm);

    void PrintSelf(ostream& os, vtkIndent indent);

    vtkSetMacro(UseNewWedges, int);
    vtkGetMacro(UseNewWedges, int);

    // Description:
    // Set the color value within each wedge volume.  Note, this value
    // will only be used if the UseSingleFrame parameter is turned off.
    // Default is 100.
    vtkSetMacro(ColorValue, short);
    vtkGetMacro(ColorValue, short);

    // Description:
    // Set radius of the sphere.  Default is 32.0.
    void SetRadius(double r);
    vtkGetMacro(Radius, double);

    // Description:
    // Set the number of subdivisions to perform on the icosahedron when creating
    // the sphere.  0 subdivions will return the initial icosahedron.
    vtkSetClampMacro(SubdivisionLevels, int, 0, 1);
    vtkGetMacro(SubdivisionLevels, int);

    // Description:
    // Set/Get the center of the sphere.
    vtkSetVector3Macro(Center, double);
    vtkGetVector3Macro(Center, double);

    // Description:
    // If on, create the output using a single vtkImageData component.  This coloring
    // results in wedges being colored 1,2,3,...,N, where N is the number of wedges
    // (or faces), in a single image structure.  Note, wegde volumes may not overlap
    // so the volumes may vary with using this scheme.  Alternatively, if off, create the
    // output using N components, where N is the number of wedges (or faces).  This
    // way, each wedge is its own component and wedge volumes are more uniform.  The
    // wedge color is then the same value for each wedge (see [Get/Set]ColorValue).
    // Default is Off.
    vtkSetClampMacro(UseSingleFrame, int, 0, 1);
    vtkGetMacro(UseSingleFrame, int);
    vtkBooleanMacro(UseSingleFrame, int);

protected:
    vtkjoColoredSphereSource();
    ~vtkjoColoredSphereSource() {};

    int UseNewWedges;
    short ColorValue;
    double Radius;
    int SubdivisionLevels;
    int WholeExtent[6];
    double Center[3];
    int UseSingleFrame;

    virtual int RequestInformation(vtkInformation*, vtkInformationVector**, vtkInformationVector*);
    virtual int RequestData(vtkInformation*, vtkInformationVector**, vtkInformationVector*);

};

#endif /* VTKJOCOLOREDSPHERE_H_ */
