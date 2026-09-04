/* Reference table of kappa-distribution fits from the upstream symphony code
 * (https://github.com/AFD-Illinois/symphony, GPL; not copied here), in ipole's sign convention
 * (Q negated). ipole's copy of kappa_fits.c cubes the Gamma(kappa/4 - 1/3) factor of kappa_I's
 * N_high (upstream has it once), so the kappa table is generated from upstream. Build in a
 * symphony checkout:
 *     gcc -std=c99 -O2 -DDEBUG=0 -I src -I src/integrator -I src/maxwell_juettner -I src/power_law -I src/kappa \
 *         -I src/susceptibility_tensor -o kappa_table kappa_table.c src/symphony.c src/params.c src/fits.c \
 *         src/distribution_function_common_routines.c src/bessel_mod.c src/integrator/*.c src/maxwell_juettner/*.c \
 *         src/power_law/*.c src/kappa/*.c src/susceptibility_tensor/*.c -lgsl -lgslcblas -lm
 *     ./kappa_table | grep "^nu,\\|^[0-9]" > kappa_table.csv    (symphony prints validity warnings to stdout)
 * Rotativities come from rho_nu_fit (Marszewski+ 2021 fits with the codes' piecewise interpolation in
 * kappa); at kappa = 5 exactly both codes divide the rho_Q interpolation by 5 instead of 3, so those
 * rho_Q values are excluded from the comparison. */
#include <stdio.h>
#include <math.h>
#include "symphony.h"
#include "params.h"
#include "fits.h"
static const double NUS[] = {86e9, 230e9, 345e9, 1e12};
static const double KAPPAS[] = {3.5, 4.0, 4.25, 4.5, 5.0};
static const double WIDTHS[] = {3.0, 10.0, 30.0};
static const double BS[] = {1.0, 10.0, 50.0, 100.0};
static const double ANGLES_DEG[] = {10.0, 30.0, 45.0, 60.0, 80.0, 89.0, 100.0, 135.0, 170.0};
#define LEN(a) (sizeof(a) / sizeof((a)[0]))
int main(void) {
  struct parameters p; setConstParams(&p);
  printf("nu,kappa,w,B,ne,theta,hyp2f1,jI,jQ,jV,aI,aQ,aV,rhoQ,rhoV\n");
  for (size_t i = 0; i < LEN(NUS); i++) for (size_t j = 0; j < LEN(KAPPAS); j++) for (size_t g = 0; g < LEN(WIDTHS); g++)
  for (size_t k = 0; k < LEN(BS); k++) for (size_t m = 0; m < LEN(ANGLES_DEG); m++) {
    double nu = NUS[i], kap = KAPPAS[j], w = WIDTHS[g], B = BS[k], ne = 1e5, th = ANGLES_DEG[m] * M_PI / 180.0;
#define FIT(f, pol) f(nu, B, ne, th, p.KAPPA_DIST, pol, 10.0, 3.0, 10.0, 1e5, 1e10, kap, w)
    double jI = FIT(j_nu_fit, p.STOKES_I), jQ = -FIT(j_nu_fit, p.STOKES_Q), jV = FIT(j_nu_fit, p.STOKES_V);
    double aI = FIT(alpha_nu_fit, p.STOKES_I), aQ = -FIT(alpha_nu_fit, p.STOKES_Q), aV = FIT(alpha_nu_fit, p.STOKES_V);
    double rQ = FIT(rho_nu_fit, p.STOKES_Q), rV = FIT(rho_nu_fit, p.STOKES_V);
    printf("%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
           nu, kap, w, B, ne, th, 0.0, jI, jQ, jV, aI, aQ, aV, rQ, rV);
  }
  return 0;
}
