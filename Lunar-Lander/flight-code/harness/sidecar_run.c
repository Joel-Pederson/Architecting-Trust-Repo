/*
 * sidecar_run.c - run the generated barrier over recorded telemetry and print what it
 * does, step by step. This is the demonstration rather than the test: it exists so the
 * barrier's behaviour can be watched outside MATLAB entirely.
 *
 * Usage:  ./sidecar_run ../fixtures/telemetry.csv
 */

#include <stdio.h>
#include <stdlib.h>

#include "safety_sidecar_filter.h"
#include "safety_sidecar_filter_initialize.h"
#include "safety_sidecar_filter_terminate.h"

int main(int argc, char **argv) {
    const char *path = (argc > 1) ? argv[1] : "../fixtures/telemetry.csv";
    FILE *f = fopen(path, "r");
    char line[4096];
    char *p, *end;
    double v[10], x[8], u[2], u_out[2], h_alt, h_fuel;
    boolean_T veto;
    long n = 0, vetoes = 0;
    int i;

    if (!f) { fprintf(stderr, "cannot open %s\n", path); return 2; }

    safety_sidecar_filter_initialize();
    printf("  step   alt(m)   dy(m/s)  theta(rad) | pilot(N)  allowed(N)  veto |  h_alt(m)\n");
    printf("  ----------------------------------------------------------------------------\n");

    while (fgets(line, sizeof line, f)) {
        p = line;
        for (i = 0; i < 10; i++) {
            v[i] = strtod(p, &end);
            if (end == p) break;
            p = (*end == ',') ? end + 1 : end;
        }
        if (i < 10) continue;

        for (i = 0; i < 8; i++) x[i] = v[i];
        u[0] = v[8]; u[1] = v[9];

        safety_sidecar_filter(x, u, u_out, &veto, &h_alt, &h_fuel);
        n++;
        if (veto) vetoes++;

        /* Print the steps where the barrier actually intervenes, plus a periodic sample
         * so the descent is legible without dumping every row. */
        if (veto || n % 25 == 1) {
            printf("  %4ld %8.1f %9.2f %11.3f | %8.0f %11.0f %5s | %9.2f\n",
                   n, x[1], x[3], x[4], u[0], u_out[0], veto ? "YES" : "-", h_alt);
        }
    }

    safety_sidecar_filter_terminate();
    fclose(f);
    printf("  ----------------------------------------------------------------------------\n");
    printf("  %ld steps, barrier engaged on %ld of them (%.1f%%)\n",
           n, vetoes, n ? 100.0 * (double)vetoes / (double)n : 0.0);
    return 0;
}
