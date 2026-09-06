/*
 * sidecar_equivalence.c - does the generated C make the same decisions as the MATLAB?
 *
 * Reads two fixtures written by flight-code/export_flight_fixtures.m:
 *
 *   telemetry.csv         x1..x8,u1,u2      one row per control step
 *   matlab_reference.csv  ua1,ua2,veto,h_alt,h_fuel   what the MATLAB barrier did
 *
 * Runs the generated barrier over the telemetry and compares. Exits non-zero on any
 * disagreement, so it works as a test as well as a demonstration.
 *
 * The veto flag is compared EXACTLY. It is a control-authority decision, not a
 * measurement: "the barrier took over" either matches or it does not, and a tolerance on
 * a boolean would hide precisely the divergence worth catching. The continuous outputs
 * are compared against a tolerance, because floating-point contraction is permitted to
 * differ between the MATLAB VM and whatever the C compiler emits.
 *
 * Build: see the Makefile in this directory. No MATLAB required.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "safety_sidecar_filter.h"
#include "safety_sidecar_filter_initialize.h"
#include "safety_sidecar_filter_terminate.h"

/* Loose enough to permit fused multiply-add and reassociation, tight enough that a real
 * behavioural change cannot hide underneath it. Commanded thrust is O(1e4) N, so this is
 * roughly one part in 1e13 of the signal. */
#define TOL 1e-6

static int read_row(FILE *f, double *v, int n) {
    char line[4096];
    char *p, *end;
    int i;

    while (fgets(line, sizeof line, f)) {
        if (line[0] == '\n' || line[0] == '#') continue;
        p = line;
        for (i = 0; i < n; i++) {
            v[i] = strtod(p, &end);
            if (end == p) return 0;               /* not a number: header row */
            p = (*end == ',') ? end + 1 : end;
        }
        return 1;
    }
    return 0;
}

int main(int argc, char **argv) {
    const char *tel_path = (argc > 1) ? argv[1] : "../fixtures/telemetry.csv";
    const char *ref_path = (argc > 2) ? argv[2] : "../fixtures/matlab_reference.csv";

    FILE *ft = fopen(tel_path, "r");
    FILE *fr = fopen(ref_path, "r");
    double row[10], ref[5];
    double x[8], u[2], u_out[2], h_alt, h_fuel;
    boolean_T veto;
    long n = 0, mismatches = 0, veto_diffs = 0, vetoes_c = 0, vetoes_m = 0;
    double worst_cmd = 0.0, worst_margin = 0.0;

    if (!ft) { fprintf(stderr, "cannot open %s\n", tel_path); return 2; }
    if (!fr) { fprintf(stderr, "cannot open %s\n", ref_path); fclose(ft); return 2; }

    safety_sidecar_filter_initialize();

    while (read_row(ft, row, 10) && read_row(fr, ref, 5)) {
        memcpy(x, row,     8 * sizeof(double));
        memcpy(u, row + 8, 2 * sizeof(double));

        safety_sidecar_filter(x, u, u_out, &veto, &h_alt, &h_fuel);
        n++;

        if (veto)          vetoes_c++;
        if (ref[2] != 0.0) vetoes_m++;

        double d_cmd = fabs(u_out[0] - ref[0]);
        if (fabs(u_out[1] - ref[1]) > d_cmd) d_cmd = fabs(u_out[1] - ref[1]);

        double d_margin = fabs(h_alt - ref[3]);
        if (fabs(h_fuel - ref[4]) > d_margin) d_margin = fabs(h_fuel - ref[4]);

        if (d_cmd    > worst_cmd)    worst_cmd    = d_cmd;
        if (d_margin > worst_margin) worst_margin = d_margin;

        int veto_bad = ((double)(veto != 0) != (ref[2] != 0.0 ? 1.0 : 0.0));
        if (veto_bad) veto_diffs++;

        if (veto_bad || d_cmd > TOL || d_margin > TOL) {
            if (mismatches < 5) {
                fprintf(stderr,
                        "  row %ld: d_cmd=%.3e d_margin=%.3e veto C=%d MATLAB=%.0f\n",
                        n, d_cmd, d_margin, (int)(veto != 0), ref[2]);
            }
            mismatches++;
        }
    }

    safety_sidecar_filter_terminate();
    fclose(ft);
    fclose(fr);

    if (n == 0) {
        fprintf(stderr, "no rows compared - are the fixtures present?\n");
        return 2;
    }

    printf("  rows compared        : %ld\n", n);
    printf("  vetoes  C / MATLAB   : %ld / %ld\n", vetoes_c, vetoes_m);
    printf("  worst |d command|    : %.3e N\n", worst_cmd);
    printf("  worst |d margin|     : %.3e m\n", worst_margin);
    printf("  veto decisions differing : %ld\n", veto_diffs);
    printf("  tolerance            : %.0e\n", TOL);

    if (mismatches) {
        printf("  RESULT: DIVERGENT (%ld rows)\n", mismatches);
        return 1;
    }
    printf("  RESULT: the generated C reproduces the MATLAB barrier\n");
    return 0;
}
