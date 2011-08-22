//BTX
/*
  Compatibility Hacks for MacOSX and SGI IRIX
*/

#ifndef PXISINF_INCLUDE
#define PXISINF_INCLUDE


#ifdef __APPLE__

  #if ( __GNUC__ >=4 ) 
    #include <cmath>
    #define isnan(x) std::isnan(x)
    #define isinf(x) std::isinf(x)
  #elif (( __GNUC__ ==3 ) && ( __GNUC_MINOR__ >3 ))
    #include  <math.h>
  #else
    extern "C" {  int isnan(double); }
    extern "C" {  int isinf(double);  }
  #endif
#endif

#ifndef __APPLE__
    #ifndef isnan
       # define isnan(x) \
           (sizeof (x) == sizeof (long double) ? isnan_ld (x) \
           : sizeof (x) == sizeof (double) ? isnan_d (x) \
           : isnan_f (x))
       static inline int isnan_f  (float       x) { return x != x; }
       static inline int isnan_d  (double      x) { return x != x; }
       static inline int isnan_ld (long double x) { return x != x; }
     #endif

     #ifndef isinf
        # define isinf(x) \
          (sizeof (x) == sizeof (long double) ? isinf_ld (x) \
           : sizeof (x) == sizeof (double) ? isinf_d (x) \
           : isinf_f (x))
       static inline int isinf_f  (float       x) { return isnan (x - x); }
       static inline int isinf_d  (double      x) { return isnan (x - x); }
       static inline int isinf_ld (long double x) { return isnan (x - x); }
     #endif
#endif

#ifndef M_PI
#define M_PI 3.1415926854
#endif

#endif /* PXISINF_INCLUDE */

//ETX
