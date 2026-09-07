# Build the Safety Sidecar barrier and its harnesses. No MATLAB required.
#
#   make          build both harnesses
#   make check    build, then verify the C reproduces the MATLAB reference (exit 1 if not)
#   make demo     build, then print what the barrier does over the recorded episode
#   make bench    build, then time a single barrier call
#   make clean
#
# The generated sources in ../src are committed, so this builds from a fresh clone with
# nothing but a C99 compiler.

CC      ?= cc
CFLAGS  ?= -O2 -std=c99 -Wall -Wextra -pedantic

# libm must be linked explicitly. The barrier calls fmax, fmin, atan2 and sqrt; on macOS
# those live in libSystem and link implicitly, but on glibc they do not, so a build that
# passed locally failed on the Linux CI runner with undefined references. An artefact
# whose claim is portability has to be built somewhere other than where it was written.
LDLIBS  ?= -lm

SRCDIR   = ../src
FIXTURES = ../fixtures
INCLUDES = -I$(SRCDIR)

GENERATED = $(SRCDIR)/safety_sidecar_filter.c \
            $(SRCDIR)/safety_sidecar_filter_initialize.c \
            $(SRCDIR)/safety_sidecar_filter_terminate.c

BINARIES = sidecar_equivalence sidecar_run sidecar_bench

.PHONY: all check demo bench clean

all: $(BINARIES)

sidecar_equivalence: sidecar_equivalence.c $(GENERATED)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDLIBS)

sidecar_run: sidecar_run.c $(GENERATED)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDLIBS)

sidecar_bench: sidecar_bench.c $(GENERATED)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDLIBS)

check: sidecar_equivalence
	@./sidecar_equivalence $(FIXTURES)/telemetry.csv $(FIXTURES)/matlab_reference.csv

demo: sidecar_run
	@./sidecar_run $(FIXTURES)/telemetry.csv

bench: sidecar_bench
	@./sidecar_bench

clean:
	rm -f $(BINARIES)
