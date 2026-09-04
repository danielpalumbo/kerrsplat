/* Reference tables for KerrSplat.Transfer from the symphony fitting formulae shipped with ipole.
 *
 * Build against a local ipole checkout (GPL; nothing from it is copied into this repository):
 *     make IPOLE_DIR=$HOME/local_scripts/ipole
 * writes thermal_table.csv and powerlaw_table.csv, which the Julia tests read.
 *
 * The rows reproduce what ipole's jar_calc_dist (model_radiation.c) computes for its default
 * thermal prescription E_DEXTER_THERMAL and for E_POWERLAW, in ipole's sign convention
 * (symphony's Q and U are negated), before the division by ν² / multiplication by ν that
 * ipole applies to form invariants, and before its polarization-fraction caps:
 *   thermal: j_{I,Q,V} from Dexter (2016) [maxwell_juettner_dexter_*], α_S = j_S / B_ν (Kirchhoff),
 *            ρ_Q from Dexter (2016) [maxwell_juettner_rho_Q], ρ_V from Shcherbakov (2008)
 *            [maxwell_juettner_rho_V with dexter_fit = 0]; also the Leung+ (2011) unpolarized j_I
 *            that ipole uses for unpolarized transport, the Pandya+ (2016) thermal fits and the
 *            Dexter ρ_V variant, for completeness.
 *   power law: j_{I,Q,V} and α_{I,Q,V} from Pandya+ (2016) [power_law_*]; symphony has no
 *            power-law rotativities.
 *   kappa:   see kappa_table.c (generated from upstream symphony, whose kappa_I is the correct one).
 */
#include <stdio.h>
#include <math.h>
#include "params.h"
#include "fits.h"
#include "constants.h"

/* ipole's radiation.c: Planck function divided by ν³ */
double Bnu_inv(double nu, double Thetae)
{
  double x = HPL * nu / (ME * CL * CL * Thetae);
  if (x < 2.e-3)
    return ((2. * HPL / (CL * CL)) / (x / 24. * (24. + x * (12. + x * (4. + x)))));
  else
    return ((2. * HPL / (CL * CL)) / (exp(x) - 1.));
}

static const double NUS[] = {86e9, 230e9, 345e9, 1e12};
static const double THETAES[] = {0.3, 0.5, 1.0, 3.0, 10.0, 30.0, 100.0};
static const double BS[] = {1.0, 10.0, 50.0, 100.0};
static const double NES[] = {1e5, 1e6};
static const double ANGLES_DEG[] = {10.0, 30.0, 45.0, 60.0, 80.0, 89.0, 100.0, 135.0, 170.0};
static const double PS[] = {2.5, 3.0, 3.5};
static const double GMINS[] = {10.0, 100.0};
#define LEN(a) (sizeof(a) / sizeof((a)[0]))

int main(void)
{
  FILE *ft = fopen("thermal_table.csv", "w");
  fprintf(ft, "nu,Thetae,B,ne,theta,jI,jQ,jV,aI,aQ,aV,rhoQ,rhoV,jI_leung,jI_pandya,jQ_pandya,jV_pandya,rhoV_dexter\n");
  for (size_t i = 0; i < LEN(NUS); i++)
  for (size_t j = 0; j < LEN(THETAES); j++)
  for (size_t k = 0; k < LEN(BS); k++)
  for (size_t l = 0; l < LEN(NES); l++)
  for (size_t m = 0; m < LEN(ANGLES_DEG); m++) {
    struct parameters p;
    setConstParams(&p);
    p.distribution = p.MAXWELL_JUETTNER;
    p.nu = NUS[i]; p.theta_e = THETAES[j]; p.magnetic_field = BS[k];
    p.electron_density = NES[l]; p.observer_angle = ANGLES_DEG[m] * M_PI / 180.0;
    p.dexter_fit = 1;
    double jI = j_nu_fit(&p, p.STOKES_I);
    double jQ = -j_nu_fit(&p, p.STOKES_Q);
    double jV = j_nu_fit(&p, p.STOKES_V);
    double Bnu = Bnu_inv(p.nu, p.theta_e) * pow(p.nu, 3);
    double aI = jI / Bnu, aQ = jQ / Bnu, aV = jV / Bnu;
    p.dexter_fit = 0;
    double rhoQ = rho_nu_fit(&p, p.STOKES_Q);
    double rhoV = rho_nu_fit(&p, p.STOKES_V);
    double jI_pandya = j_nu_fit(&p, p.STOKES_I);
    double jQ_pandya = -j_nu_fit(&p, p.STOKES_Q);
    double jV_pandya = j_nu_fit(&p, p.STOKES_V);
    p.dexter_fit = 2;
    double jI_leung = j_nu_fit(&p, p.STOKES_I);
    p.dexter_fit = 1;
    double rhoV_dexter = rho_nu_fit(&p, p.STOKES_V);
    fprintf(ft, "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
            p.nu, p.theta_e, p.magnetic_field, p.electron_density, p.observer_angle,
            jI, jQ, jV, aI, aQ, aV, rhoQ, rhoV, jI_leung, jI_pandya, jQ_pandya, jV_pandya, rhoV_dexter);
  }
  fclose(ft);

  FILE *fp = fopen("powerlaw_table.csv", "w");
  fprintf(fp, "nu,p,gamma_min,gamma_max,B,ne,theta,jI,jQ,jV,aI,aQ,aV\n");
  for (size_t i = 0; i < LEN(NUS); i++)
  for (size_t j = 0; j < LEN(PS); j++)
  for (size_t g = 0; g < LEN(GMINS); g++)
  for (size_t k = 0; k < LEN(BS); k++)
  for (size_t m = 0; m < LEN(ANGLES_DEG); m++) {
    struct parameters p;
    setConstParams(&p);
    p.distribution = p.POWER_LAW;
    p.nu = NUS[i]; p.power_law_p = PS[j]; p.gamma_min = GMINS[g]; p.gamma_max = 1e5; p.gamma_cutoff = 1e10;
    p.magnetic_field = BS[k]; p.electron_density = 1e5; p.observer_angle = ANGLES_DEG[m] * M_PI / 180.0;
    double jI = j_nu_fit(&p, p.STOKES_I);
    double jQ = -j_nu_fit(&p, p.STOKES_Q);
    double jV = j_nu_fit(&p, p.STOKES_V);
    double aI = alpha_nu_fit(&p, p.STOKES_I);
    double aQ = -alpha_nu_fit(&p, p.STOKES_Q);
    double aV = alpha_nu_fit(&p, p.STOKES_V);
    fprintf(fp, "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
            p.nu, p.power_law_p, p.gamma_min, p.gamma_max, p.magnetic_field, p.electron_density, p.observer_angle,
            jI, jQ, jV, aI, aQ, aV);
  }
  fclose(fp);

  return 0;
}
