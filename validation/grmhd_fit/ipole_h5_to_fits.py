"""Convert an ipole HDF5 image (the `pol` dataset: I, Q, U, V, tau in cgs specific intensity) to an
ehtim-style Stokes FITS file (Jy/pixel, one HDU per Stokes parameter) that Fit.read_stokes_fits reads.
Usage: python ipole_h5_to_fits.py <in.h5> <out.fits>; prints the geometry of the run."""
import sys, h5py, numpy as np
from astropy.io import fits
f = h5py.File(sys.argv[1], 'r'); h = f['header']
pol = f['pol'][()]                                     # (nx, ny, 5), x index first, y index second
scale = h['scale'][()]                                 # pixel solid angle / Jy
nx, ny = pol.shape[:2]
fov_uas = h['camera']['fovx_dsource'][()]; psize_rad = fov_uas * 1e-6 / 206264.806247 / nx
freq = h['freqcgs'][()]; dsource = h['dsource'][()]; L = h['units']['L_unit'][()]
G, c, Msun = 6.6743e-8, 2.99792458e10, 1.989e33
M_solar = L * c**2 / G / Msun
hdus = []
for k, st in enumerate('IQUV'):
    img = (pol[:, :, k] * scale).T                     # FITS axis 1 = x (increasing toward the west, ipole's CDELT1 < 0 layout), axis 2 = y (north)
    hd = fits.PrimaryHDU(img) if k == 0 else fits.ImageHDU(img)
    hd.header['OBJECT'] = 'GRMHD'; hd.header['CTYPE1'] = 'RA---SIN'; hd.header['CTYPE2'] = 'DEC--SIN'
    hd.header['CDELT1'] = -np.degrees(psize_rad); hd.header['CDELT2'] = np.degrees(psize_rad)
    hd.header['CRPIX1'] = nx / 2 + 0.5; hd.header['CRPIX2'] = ny / 2 + 0.5; hd.header['CRVAL1'] = 0.0; hd.header['CRVAL2'] = 0.0
    hd.header['CUNIT1'] = 'deg'; hd.header['CUNIT2'] = 'deg'; hd.header['FREQ'] = float(freq); hd.header['MJD'] = 58000.0
    hd.header['BUNIT'] = 'JY/PIXEL'; hd.header['STOKES'] = st; hd.header['TELESCOP'] = 'ipole'
    hdus.append(hd)
fits.HDUList(hdus).writeto(sys.argv[2], overwrite=True)
print('nx', nx, 'ny', ny, 'fov_uas', fov_uas, 'psize_uas', fov_uas / nx, 'freq', freq, 'M_solar', M_solar, 'D_pc', dsource / 3.0857e18,
      'a', f['fluid_header']['a'][()], 'thetacam', h['camera']['thetacam'][()], 'Ftot', f['Ftot'][()], 'sum I', pol[:, :, 0].sum() * scale,
      'sum Q,U,V', pol[:, :, 1].sum() * scale, pol[:, :, 2].sum() * scale, pol[:, :, 3].sum() * scale, 'evpa_0', h['evpa_0'][()])
