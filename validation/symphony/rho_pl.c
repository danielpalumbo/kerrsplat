/* Power-law rotativities from symphony's numerical susceptibility-tensor integration (the full
 * symphony code, https://github.com/AFD-Illinois/symphony, GPL; not copied here), for a small
 * grid; writes powerlaw_rho_table.csv, the reference of test_powerlaw_rotativities. Build in a
 * symphony checkout (GSL needed; each rho_nu call takes minutes, so run one process per
 * (frequency, field) pair with the two optional arguments and concatenate the outputs):
 *     gcc -std=c99 -O2 -DDEBUG=0 -I src -I src/integrator -I src/maxwell_juettner -I src/power_law -I src/kappa \
 *         -I src/susceptibility_tensor -o rho_pl rho_pl.c src/symphony.c src/params.c src/fits.c \
 *         src/distribution_function_common_routines.c src/bessel_mod.c src/integrator/*.c src/maxwell_juettner/*.c \
 *         src/power_law/*.c src/kappa/*.c src/susceptibility_tensor/*.c -lgsl -lgslcblas -lm
 *     ./rho_pl 230e9 10 > part.csv      (or ./rho_pl for the whole grid) */
#include <stdio.h>
#include <math.h>
#include "symphony.h"
#include "params.h"
#include <stdlib.h>
int main(int argc, char **argv) {
  struct parameters p; setConstParams(&p);
  /* optional arguments: a single frequency and field strength (parallel runs), else the full grid */
  double nus[] = {86e9, 230e9, 345e9};
  double Bs[] = {10.0, 50.0};
  int nnu = 3, nB = 2;
  if (argc == 3) { nus[0] = atof(argv[1]); Bs[0] = atof(argv[2]); nnu = 1; nB = 1; }
  double thetas[] = {30.0, 60.0, 80.0};
  double ps[] = {2.5, 3.0, 3.5};
  double gmins[] = {10.0, 100.0};
  printf("nu,B,theta,p,gamma_min,gamma_max,ne,rhoQ,rhoV\n");
  for (int i = 0; i < nnu; i++) for (int j = 0; j < nB; j++) for (int k = 0; k < 3; k++) for (int l = 0; l < 3; l++) for (int m = 0; m < 2; m++) {
    double nu = nus[i], B = Bs[j], th = thetas[k] * M_PI / 180., pp = ps[l], gmin = gmins[m], gmax = 1e5, ne = 1e5;
    char *err = NULL;
    double rq = rho_nu(nu, B, ne, th, p.POWER_LAW, p.STOKES_Q, 10., pp, gmin, gmax, 1e10, 3.5, 10., &err);
    double rv = rho_nu(nu, B, ne, th, p.POWER_LAW, p.STOKES_V, 10., pp, gmin, gmax, 1e10, 3.5, 10., &err);
    printf("%.10g,%.10g,%.17g,%.3g,%.10g,%.10g,%.10g,%.17g,%.17g\n", nu, B, th, pp, gmin, gmax, ne, rq, rv);
    fflush(stdout);
  }
  return 0;
}
