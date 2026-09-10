"""ehtim's Jones corruption of the noiseless synthetic EHT 2017 observation, as fixtures for the instrument-model gate:
the clean and corrupted circular products per row, and the Jones matrix ehtim applied per station and time, for the
feed-rotation-corrected case (frcal=True: leakage rotated by twice the feed angle, Comrade's R† G D R) and the raw case
(frcal=False: G D R), with the feed rotation angle recomputed from the array table. Seeded, so the draws are reproducible.

    python dump_jones.py <noiseless.uvfits> <EHT2017.txt> <outdir>
"""
import sys, numpy as np, ehtim as eh
import ehtim.observing.obs_simulate as sim, ehtim.observing.obs_helpers as obsh, ehtim.const_def as ehc
obs = eh.obsdata.load_uvfits(sys.argv[1])
arr = eh.array.load_txt(sys.argv[2])
# the array table's mount parameters (fr_par, fr_elev, fr_off) for the observed stations, matched by position (the
# uvfits carries two-letter codes and no mount information)
for i in range(len(obs.tarr)):
    xyz = np.array([obs.tarr[i]['x'], obs.tarr[i]['y'], obs.tarr[i]['z']])
    d = [np.linalg.norm(xyz - np.array([r['x'], r['y'], r['z']])) for r in arr.tarr]
    k = int(np.argmin(d)); assert d[k] < 1e3, (obs.tarr[i]['site'], d[k])
    for f in ('fr_par', 'fr_elev', 'fr_off', 'sefdr', 'sefdl'):
        obs.tarr[i][f] = arr.tarr[k][f]
    print(obs.tarr[i]['site'], '->', arr.tarr[k]['site'], 'fr', arr.tarr[k]['fr_par'], arr.tarr[k]['fr_elev'], arr.tarr[k]['fr_off'])
circ = obs.switch_polrep('circ')
opts = dict(add_th_noise=False, opacitycal=True, ampcal=False, phasecal=False, dcal=False, rlgaincal=False,
            gainp=0.1, gain_offset=0.1, phase_std=-1, dterm_offset=0.1, rlratio_std=0.1, rlphase_std=0.1, seed=42, verbose=False)
for label, frcal in (("corrected", True), ("raw", False)):
    data = sim.add_jones_and_noise(obs, frcal=frcal, **opts)
    jm = sim.make_jones(obs, opacitycal=True, ampcal=False, phasecal=False, dcal=False, frcal=frcal, rlgaincal=False,
                        gainp=0.1, gain_offset=0.1, phase_std=-1, dterm_offset=0.1, rlratio_std=0.1, rlphase_std=0.1, seed=42)
    cor = obs.copy(); cor.data = data; cor = cor.switch_polrep('circ')
    rows = []
    for a, b in zip(circ.data, cor.data):
        assert a['time'] == b['time'] and a['t1'] == b['t1'] and a['t2'] == b['t2']
        rows.append('%.9g,%s,%s,' % (a['time'], a['t1'], a['t2']) + ','.join('%.12g,%.12g' % (x.real, x.imag) for x in
                    (a['rrvis'], a['llvis'], a['rlvis'], a['lrvis'], b['rrvis'], b['llvis'], b['rlvis'], b['lrvis'])))
    open('%s/jones_%s_rows.csv' % (sys.argv[3], label), 'w').write(
        'time_h,t1,t2,RRre,RRim,LLre,LLim,RLre,RLim,LRre,LRim,cRRre,cRRim,cLLre,cLLim,cRLre,cRLim,cLRre,cLRim\n' + '\n'.join(rows) + '\n')
    # the Jones matrices and the feed rotation angle of every (station, time) ehtim used
    times = np.array(sorted(set(circ.data['time'])))
    sidereal = obsh.utc_to_gmst(times, obs.mjd)
    sourcevec = np.array([np.cos(obs.dec * ehc.DEGREE), 0, np.sin(obs.dec * ehc.DEGREE)])
    lines = []
    for i in range(len(obs.tarr)):
        site = obs.tarr[i]['site']
        coords = np.array([obs.tarr[i]['x'], obs.tarr[i]['y'], obs.tarr[i]['z']])
        latlon = obsh.xyz_2_latlong(coords)
        el = obsh.elev(obsh.earthrot(coords, np.mod((sidereal - obs.ra) * ehc.HOUR, 2 * np.pi)), sourcevec)
        par = obsh.par_angle(obsh.hr_angle(sidereal * ehc.HOUR, latlon[:, 1], obs.ra * ehc.HOUR), latlon[:, 0], obs.dec * ehc.DEGREE)
        phi = obs.tarr[i]['fr_elev'] * el + obs.tarr[i]['fr_par'] * par + obs.tarr[i]['fr_off'] * ehc.DEGREE
        for j, t in enumerate(times):
            J = jm[site][t]
            lines.append('%s,%.9g,%.12g,%.12g,%.12g,' % (site, t, phi[j], el[j], par[j]) + ','.join('%.12g,%.12g' % (x.real, x.imag) for x in (J[0][0], J[0][1], J[1][0], J[1][1])))
    open('%s/jones_%s_matrices.csv' % (sys.argv[3], label), 'w').write('site,time_h,phi,elev,parang,J11re,J11im,J12re,J12im,J21re,J21im,J22re,J22im\n' + '\n'.join(lines) + '\n')
    print(label, len(rows), 'rows,', len(lines), 'station-times')
# the station table the angles came from
open('%s/jones_stations.csv' % sys.argv[3], 'w').write('site,x,y,z,fr_par,fr_elev,fr_off_deg\n' + '\n'.join(
    '%s,%.6f,%.6f,%.6f,%g,%g,%g' % (r['site'], r['x'], r['y'], r['z'], r['fr_par'], r['fr_elev'], r['fr_off']) for r in obs.tarr) + '\n')
print('ra', obs.ra, 'dec', obs.dec, 'mjd', obs.mjd)
