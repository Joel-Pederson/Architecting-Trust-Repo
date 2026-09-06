/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: safety_sidecar_filter.h
 *
 * MATLAB Coder version            : 25.1
 * C/C++ source code generated on  : 06-Sep-2026 19:16:29
 */

#ifndef SAFETY_SIDECAR_FILTER_H
#define SAFETY_SIDECAR_FILTER_H

/* Include Files */
#include "rtwtypes.h"
#include <stddef.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Function Declarations */
extern void safety_sidecar_filter(const double x[8], const double u_nominal[2],
                                  double u_actual[2], boolean_T *VetoTriggered,
                                  double *h_alt, double *h_fuel);

#ifdef __cplusplus
}
#endif

#endif
/*
 * File trailer for safety_sidecar_filter.h
 *
 * [EOF]
 */
