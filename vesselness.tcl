#!/bin/sh
# the next line restarts using wish \
    exec vtk "$0" -- "$@"

lappend auto_path [ file dirname [ info script ]]

package require vtkbase
package require vtkcommon
package require vtkfiltering
package require vtkgraphics
package require vtkhybrid
package require vtkimaging
package require vtkio
package require vtkparallel
package require vtkrendering

set TIMING_MODE 0
set DOUBLE_PRECISION 0

if { $tcl_platform(platform) == "windows" } { 
	load "lib/Release/VesselnessTCL.dll"
} else {
	load "lib/libVesselnessTCL.so"
}

set argc [ llength $argv ]

if { $argc < 1 } {
    set scriptname [ file tail $argv0 ]
    puts stdout "Syntax: $scriptname input \[ beta = 800.0 \] \[ radius = 8 \] \[ usenewsphere = 0 \] \[ usenewwedges = 0 \] \[ forcecpu = 0 \] \[ outstem = \"\" \] \[ debug = 0 \]\n"
    puts stdout "\nThis computes the entropy-based vesselness images, as described in the accompanying paper."
    puts stdout "\n"
    exit 0
}

set name [ lindex $argv 0 ]
set stem [string range $name 0 [expr [string last "." $name]-1]]

set params [ list 800.0 8.0 0 0 0 $stem 0 ]

for {set i 1} {$i<$argc} {incr i} {
    lset params [expr $i-1] [lindex $argv $i]
}

set beta         [lindex $params 0]
set radius       [lindex $params 1]
set usenewsphere [lindex $params 2]
set usenewwedges [lindex $params 3]
set forcecpu     [lindex $params 4]
set outputstem   [lindex $params 5]
set debug        [lindex $params 6]

#note: the origin is not where you'd expect it to be with StructuredPointsReader!
set reader [vtkStructuredPointsReader New]
$reader SetFileName $name
$reader Update

set image_in [ $reader GetOutput ]
#$image_in SetOrigin 0 0 0

set castimage [vtkImageCast New]
$castimage SetInput $image_in
if {$DOUBLE_PRECISION==1} {
    $castimage SetOutputScalarTypeToDouble
} else {
    $castimage SetOutputScalarTypeToFloat
}
$castimage Update

set image_in [ $castimage GetOutput ]

if {$debug==1} {
    puts "Origin = [ $image_in GetOrigin ]"
}

set timer [clock seconds]

set range [ [ [ $image_in GetPointData ] GetScalars ] GetRange ]
set K [expr 1.0/([lindex $range 1]-[lindex $range 0])]

set conv [ vtkNewImageConvolution New ]

set filtertimer [clock seconds]

if { $usenewsphere == 0 } {
    set filterbank [ $conv qVesselFilter 90.0 [ expr int($radius) ] 0.0 0.0 ]
} else {
    set sph [ vtkjoColoredSphereSource New ]
    $sph SetRadius [ expr int($radius) ]
    $sph SetColorValue 1
    $sph SetSubdivisionLevels 0
    $sph SetUseNewWedges $usenewwedges
    $sph Update
    set filterbank [ vtkImageData New ]
    $filterbank ShallowCopy [ $sph GetOutput ]
    $sph Delete
}

if {$TIMING_MODE==1} {
    puts "Filters done in [expr ([clock seconds] - $filtertimer)/1.0] seconds."
}

set convtimer [clock seconds]

set castspheres [vtkImageCast New]
if {$DOUBLE_PRECISION} {
    $castspheres SetOutputScalarTypeToDouble
} else {
    $castspheres SetOutputScalarTypeToFloat
}
$castspheres SetInput $filterbank
$castspheres Update

$conv SetInput  $image_in
$conv SetFilterBank [ $castspheres GetOutput]
$conv SetMode 2
$conv SetDoublePrecision $DOUBLE_PRECISION
$conv SetDebug $debug
$conv SetForceCPU $forcecpu
$conv SetDoComplexFFTs 1
$conv SetTimingMode $TIMING_MODE
$conv Update

set castconv [vtkImageCast New]
$castconv SetInput [$conv GetOutput]
if {$DOUBLE_PRECISION} {
    $castconv SetOutputScalarTypeToDouble
} else {
    $castconv SetOutputScalarTypeToFloat
}
$castconv Update

if {$TIMING_MODE==1} {
    puts "Convolutions done in [expr ([clock seconds] - $convtimer)/1.0] seconds."
    puts "\nStarting v(x) calculations.\n"
}
set ttimer [clock seconds]

set tightness [ vtkNewODFVesselnessFilter New ]
$tightness SetInput [ $castconv GetOutput ]
$tightness SetBeta [ expr $beta/1.0]
$tightness SetForceCPU $forcecpu
$tightness SetDebug 0
$tightness SetTimingMode $TIMING_MODE
$tightness Update

set tightnesstodouble [vtkImageCast New]
$tightnesstodouble SetInput [$tightness GetOutput]
if {$DOUBLE_PRECISION} {
    $tightnesstodouble SetOutputScalarTypeToDouble
} else {
    #misnomer? :)
    $tightnesstodouble SetOutputScalarTypeToFloat
}
$tightnesstodouble Update

if {$TIMING_MODE==1} {
    puts "\nv(x) done in [expr ([clock seconds] - $ttimer)/1.0] seconds.\n"
    puts "\nStarting b(x) calculations.\n"
}
set btimer [clock seconds]

set normalizedimage [vtkImageMathematics New]
$normalizedimage SetInput $image_in
$normalizedimage SetOperationToMultiplyByK
$normalizedimage SetConstantK $K
$normalizedimage Update

set brightness [vtkImageGaussianSmooth New]
$brightness SetInput [$normalizedimage GetOutput]
$brightness Update
#note: $brightness GetOutput is now an image consisting of our newly-defined b(x).

if {$TIMING_MODE==1} {
    puts "\nb(x) done in [expr ([clock seconds] - $btimer)/1.0] seconds."
    puts "\nStarting PPV(x) calculations...\n"
}

set ppvtimer [clock seconds]

set ppv [vtkImageMathematics New]
$ppv SetInput1 [$tightnesstodouble GetOutput]
$ppv SetInput2 [$brightness GetOutput] 
$ppv SetOperationToMultiply
$ppv Update
#note: $ppv GetOutput is now an image consisting of PPV(x)=v(x)*b(x), as defined in the paper.

if {$TIMING_MODE==1} {
    puts "\nPPV(x) done in [expr ([clock seconds] - $ppvtimer)/1.0] seconds."
    puts "Everything done in [expr ([clock seconds] - $timer)/1.0] seconds."
}

set saver [vtkStructuredPointsWriter New]
$saver SetInput [$ppv GetOutput]
$saver SetFileName "${outputstem}_vesselness.vt"
$saver Update

puts "Saved to ${outputstem}_vesselness.vt"

$tightness Delete;    $tightnesstodouble Delete;
$castimage Delete;    $normalizedimage Delete;    $brightness Delete
$ppv Delete;          $castspheres Delete;        $conv Delete
$saver Delete;        $reader Delete;             $castconv Delete

exit