cmake_minimum_required(VERSION 2.6)

# create the vtk executable
PROJECT(Vesselness)

SET(KITBASE Vesselness)
SET(KIT     ${KITBASE})
SET(CUDAKIT ${KITBASE}_Cuda)
OPTION(USE_CUDA
"Use CUDA Directories"
ON)

INCLUDE (Cmake_Modules/FindCuda.cmake)
INCLUDE (Cmake_Modules/FindVTK.cmake)
FIND_PACKAGE(VTK REQUIRED)
IF (USE_VTK_FILE)
  INCLUDE(${USE_VTK_FILE})
ENDIF(USE_VTK_FILE)


INCLUDE_DIRECTORIES(/usr/local/cuda/include)
INCLUDE_DIRECTORIES(${Vesselness_SOURCE_DIR})
# Set the default location for outputting the library
SET (LIBRARY_OUTPUT_PATH ${Vesselness_SOURCE_DIR}/lib)


SET(LIBRARY_SRCS
vtkjoColoredSphereSource.cpp
vtkjoSphereSource.cxx
vtkNewImageConvolution.cpp
vtkNewCUDAFFT.cpp
vtkNewODFVesselnessFilter.cpp
)

CUDA_INCLUDE_DIRECTORIES(
${CMAKE_CURRENT_SOURCE_DIR}
)

CUDA_ADD_LIBRARY(${CUDAKIT} STATIC
NewCUDAFFT.cu
#NewCUDAEntropy.cu
)

SET_SOURCE_FILES_PROPERTIES(
NewCUDAFFT.cu
#NewCUDAEntropy.cu
WRAP_EXCLUDE
)

SET(vtkLocal_LIBS
cudart
cufft
)

LINK_LIBRARIES(
vtkCommon
vtkCommonTCL
vtkImaging
vtkImagingTCL
vtkGraphics
vtkGraphicsTCL
vtkHybrid
vtkHybridTCL
cudart
cufft
${CUDAKIT}
)

ADD_LIBRARY(${KIT} STATIC ${LIBRARY_SRCS})

install (TARGETS ${CUDAKIT} DESTINATION lib)

INCLUDE(${VTK_CMAKE_DIR}/vtkWrapTcl.cmake)

VTK_WRAP_TCL2 (${KIT}TCL LIBRARY_TCL_SRCS ${LIBRARY_SRCS})
ADD_LIBRARY (${KIT}TCL SHARED ${LIBRARY_TCL_SRCS} ${LIBRARY_SRCS})

SET (KITEXTRAFILES
NewCUDAFFT.cu
#NewCUDAEntropy.cu
)

