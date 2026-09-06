/*
 * rtwtypes.h - portable replacement, written by flight-code/generate_flight_code.m.
 *
 * MATLAB Coder's own rtwtypes.h ends in #include "tmwtypes.h", a MathWorks header
 * that lives outside this repository. Generation uses plain C built-in types, so the
 * barrier itself needs only boolean_T; the other two are defined for completeness. That
 * lets a reader with a C compiler and no MATLAB licence build and audit this code.
 *
 * Regenerate with: generate_flight_code
 */

#ifndef RTWTYPES_H
#define RTWTYPES_H

#include <stdint.h>

typedef double        real_T;
typedef unsigned char boolean_T;
typedef int32_t       int32_T;

#ifndef true
#define true  1
#endif
#ifndef false
#define false 0
#endif

#endif /* RTWTYPES_H */
