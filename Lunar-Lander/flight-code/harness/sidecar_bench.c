/*
 * sidecar_bench.c - how long one barrier evaluation takes.
 *
 * Throughput is not the point: the barrier runs at 10 Hz against a 100 ms budget, so it
 * could be four orders of magnitude slower and still fit. The number matters because
 * BOUNDED execution time is a verification property. A component that cannot overrun its
 * slot is one fewer thing a safety case has to argue about.
 *
 * The mean below is meaningful. The maximum is NOT a worst-case execution time - this is
 * a preemptible userspace process, so the tail measures the operating system's scheduler,
 * not this function. A real WCET needs static analysis or a bare-metal target.
 */

/* clock_gettime and CLOCK_MONOTONIC are POSIX, not ISO C. Under strict -std=c99 glibc
 * hides them unless this is declared BEFORE any header is included, which builds fine on
 * macOS and fails on Linux - the same asymmetry that hid the missing -lm. */
#define _POSIX_C_SOURCE 199309L

#include <stdio.h>
#include <time.h>

#include "safety_sidecar_filter.h"
#include "safety_sidecar_filter_initialize.h"
#include "safety_sidecar_filter_terminate.h"

#define REPS 2000000L

int main(void) {
    /* A state inside the blending zone, so the barrier does real work rather than
     * short-circuiting through the pass-through path. */
    double x[8] = {0.0, 120.0, 3.0, -14.0, 0.15, 0.0, 5200.0, 220.0};
    double u[2] = {20000.0, 0.0};
    double u_out[2], h_alt, h_fuel;
    boolean_T veto;
    struct timespec t0, t1;
    long i;

    safety_sidecar_filter_initialize();

    for (i = 0; i < 10000; i++)                     /* warm the caches */
        safety_sidecar_filter(x, u, u_out, &veto, &h_alt, &h_fuel);

    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (i = 0; i < REPS; i++)
        safety_sidecar_filter(x, u, u_out, &veto, &h_alt, &h_fuel);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    safety_sidecar_filter_terminate();

    double ns = ((double)(t1.tv_sec - t0.tv_sec) * 1e9
                 + (double)(t1.tv_nsec - t0.tv_nsec)) / (double)REPS;

    printf("  mean            : %.1f ns per barrier evaluation\n", ns);
    printf("  control period  : 100,000,000 ns (10 Hz)\n");
    printf("  headroom        : %.0fx\n", 1e8 / ns);
    printf("\n  (mean over %ld calls; the tail of a userspace timing loop measures the\n", REPS);
    printf("   OS scheduler, not this function, so no WCET is claimed here.)\n");
    return 0;
}
