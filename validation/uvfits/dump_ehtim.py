"""ehtim's parse of a uvfits file as CSV (time in hours, station names, u, v in wavelengths, Stokes visibilities and noise in Jy)."""
import sys, ehtim as eh
o = eh.obsdata.load_uvfits(sys.argv[1])
rows = ['%.9g,%s,%s,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g' % (
    r['time'], r['t1'], r['t2'], r['u'], r['v'], r['vis'].real, r['vis'].imag, r['qvis'].real, r['qvis'].imag,
    r['uvis'].real, r['uvis'].imag, r['vvis'].real, r['vvis'].imag, r['sigma'], r['qsigma'], r['usigma'], r['vsigma'], r['tint']) for r in o.data]
open(sys.argv[2], 'w').write('time_h,t1,t2,u,v,Ire,Iim,Qre,Qim,Ure,Uim,Vre,Vim,sigI,sigQ,sigU,sigV,tint\n' + '\n'.join(rows) + '\n')
print(len(o.data), 'rows; rf', o.rf, 'mjd', o.mjd, 'polrep', o.polrep)
