# Phase 3: slow light and time dependence — status (2026-09-03)

Slow light is built into the geodesic layer (the regularized lookback time t̃ of every sample)
and every consumer evaluates the plasma at the emission time t_obs − t̃; the temporal envelope
of the splats makes emission appear and disappear. Convention check (test/test_slowlight.jl):
Krang's t̃ grows inward along a ray (it is the elapsed time to the observer minus the divergent
r_obs + 2 ln r_obs, negative for the outer parts of a ray and positive near the horizon), so a
flare at t_f on a ray sample with lookback t̃ is observed at t_obs = t_f + t̃.

## Gate 5 (flaring hotspot)

A splat of scale 0.6 M at r = 6 M flares with a 1.5 M envelope (a → 0, θo = 20°). In the movie
the direct image peaks at t_obs = −4.5 M against −4.73 from the t̃ of the ray crossing the splat,
and the n = 1 echo on the far side of the ring peaks at 20.0 M against 19.86; the delay is 24.5 M
(3√3π ≈ 16 M is the asymptotic photon-ring value for a source near the ring; a source at r = 6
lags more). The echo's surface brightness is as high as the direct image's (Liouville) but its
region-integrated flux is 8% of it (demagnification). A wide envelope makes the image independent
of the observation time to 1e-13 (the fast-light limit). Runs on CPU and CUDA.

## Remaining Phase 3 items

- Motion modes for the splat centres (addendum §6.2, modes A/B/C): advected centres from a
  velocity model and trajectory knots, with the pattern-vs-fluid separation test.
- Multi-frequency cubes and movie assembly over (t_obs, ν) sharing the geodesic cache (the loop
  is trivial; the deliverable is the cube layout and its likelihood).
