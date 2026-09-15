# The triband ngEHT campaign on M87* with ngehtsim (Pesce et al.): the Phase-2 reference array of Doeleman et al. (2023)
# as ngehtsim's `ngEHT` preset, each station observing the bands its receiver table assigns it, one uvfits per band and day
# (the (u, v) sampling and thermal noise of every scan; the source itself is a placeholder Gaussian whose visibilities the
# fits never see: `Fit.coverage` lends the sampling to the synthetic slow-light movie). Usage:
#     python make_campaign.py [--days 5] [--start 2026-04-01] [--weather good] [--bands 86,230,345] [--dt 24] [--tint 600] [--trest 1200] [--fpt 600]
# --fpt T lets the fringe finder integrate coherently for T seconds at the bands above 300 GHz, the proxy for the frequency phase
# transfer the reference-array paper assumes for 345 GHz detections (ngehtsim's own multi-frequency FPT path breaks when the
# stations differ per band); with the scan length (600 s) the 345 GHz array grows from 9 stations and 250 visibilities a day
# to 10 and 480, reaching 11.7 Gλ. 0 turns it off (the default fringe finder: SNR 5 over 10 s).
import argparse, datetime, os, warnings
import numpy as np
import ehtim as eh
import ngehtsim.obs.obs_generator as og
warnings.filterwarnings('ignore')
ap = argparse.ArgumentParser()
ap.add_argument('--days', type=int, default=5); ap.add_argument('--start', default='2026-04-01'); ap.add_argument('--weather', default='good')
ap.add_argument('--bands', default='86,230,345'); ap.add_argument('--dt', type=float, default=24.0); ap.add_argument('--tint', type=float, default=600.0)
ap.add_argument('--trest', type=float, default=1200.0); ap.add_argument('--bandwidth', type=float, default=8.0); ap.add_argument('--seed', type=int, default=1)
ap.add_argument('--fpt', type=float, default=600.0)
ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output'))
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
start = datetime.date.fromisoformat(args.start)
bands = [float(b) for b in args.bands.split(',')]
# a placeholder source with the M87 coordinates: a 40 μas Gaussian of 0.6 Jy with zero polarization (ehtim needs the four Stokes)
im = eh.image.make_empty(64, 200 * eh.RADPERUAS, ra=12.51373, dec=12.39112, rf=230e9, source='M87')
im = im.add_gauss(0.6, [40 * eh.RADPERUAS, 40 * eh.RADPERUAS, 0.0, 0.0, 0.0])
z = np.zeros((im.ydim, im.xdim)); im.add_qu(z, z); im.add_v(z)
summary = []
for day in range(args.days):
    date = start + datetime.timedelta(days=day)
    for f in bands:
        settings = dict(source='M87', frequency=f, bandwidth=args.bandwidth, month=date.strftime('%b'), year=date.year, day=date.day,
                        t_start=0.0, dt=args.dt, t_int=args.tint, t_rest=args.trest, weather=args.weather, ttype='fast', random_seed=args.seed + day)
        if f > 300 and args.fpt > 0:
            settings['fringe_finder'] = ['fringegroups', [5.0, args.fpt]]
        gen = og.obs_generator(settings=settings, array='ngEHT', verbosity=0)
        imf = im.copy(); imf.rf = f * 1e9
        obs = gen.make_obs(imf, addnoise=True, addgains=False, addleakage=False, flagday=False)
        obs = obs.switch_polrep('stokes')
        d = obs.data
        sites = sorted(set(d['t1']) | set(d['t2']))
        path = os.path.join(args.out, f'ngeht_M87_{date.isoformat()}_{f:.0f}GHz.uvfits')
        obs.save_uvfits(path)
        line = (f'{date.isoformat()} {f:>4.0f} GHz: {len(sites):2d} stations, {len(d):5d} visibilities, {len(np.unique(d["time"])):3d} times, '
                f'median sigma {np.median(d["sigma"])*1e3:5.1f} mJy, max |uv| {np.max(np.hypot(d["u"], d["v"]))/1e9:4.1f} Gλ, stations {sites}')
        print(line); summary.append(line)
with open(os.path.join(args.out, 'campaign_summary.txt'), 'w') as fh:
    fh.write(f'ngehtsim ngEHT preset, M87, weather {args.weather}, {args.days} days from {args.start}, bands {args.bands} GHz, {args.tint:.0f} s scans every {args.tint + args.trest:.0f} s, FPT proxy {args.fpt:.0f} s above 300 GHz\n')
    fh.write('\n'.join(summary) + '\n')
