# Fire Spotting

Fire spotting is the ignition of new fires ahead of the main front by
wind-carried embers. PROPAGATOR implements the stochastic ember-transport
formulation of Alexandridis et al. (see the [Bibliography](bibliography.md)),
and tracks where embers are generated and where they land in every
realization.

## Enabling Spotting

Spotting is controlled by two switches:

1. **The simulation flag.** Set `"do_spotting": true` in the JSON
   configuration (CLI runs), or pass `do_spotting=True` when constructing the
   `Propagator` programmatically. When the flag is off, spotting is disabled
   for every fuel regardless of the fuel definitions.
2. **The per-fuel flag.** Each fuel type in the fuel system declares whether
   it is prone to spotting. In YAML fuel definitions this is the optional
   `spotting: true` attribute; the legacy fuel system enables it for the
   `conifers` fuel only. Fuels can also tune the ignition probability of
   landing embers with the `prob_ign_by_embers` attribute. Only burning cells
   whose fuel has `spotting: true` can launch embers.

```python
from propagator.core import FUEL_SYSTEM_LEGACY, Propagator

sim = Propagator(
    dem=dem,
    veg=veg,
    fuels=FUEL_SYSTEM_LEGACY,
    do_spotting=True,
    realizations=100,
)
```

## Model

When a fire-prone cell burns, embers may be launched downwind. For each ember:

- The base travel distance (the ember's "main thrust") is sampled from a
  Gaussian distribution with mean 100 m and standard deviation 25 m.
- The distance is modulated by wind speed and by the alignment between the
  ember trajectory and the wind direction, decaying exponentially for
  trajectories opposing the wind.
- The ember lands after a travel time proportional to distance over wind
  speed, and ignites the landing cell with a constant base probability
  corrected by the receiving cell's vegetation.
- Once an ember successfully ignites a cell,
  the model adds a **delay before the spot fire becomes capable of
  propagation**, to represent the time an ember needs to smoulder and develop
  into a spreading fire. This delay is sampled per ember from a lognormal
  distribution (median 600 s, log-scale standard deviation 0.4) and added to
  the ember travel time. The cell only starts spreading once this combined time has elapsed.

With no wind, embers travel a negligible distance: spotting is effectively a
strong-wind phenomenon in this formulation.

## Outputs

When spotting is enabled, two additional per-cell fields are tracked per
realization and aggregated in every `PropagatorOutput` snapshot (and written
as GeoTIFFs by the CLI):

- `spotting_generation_probability` — fraction of realizations in which the
  cell launched at least one ember.
- `spotting_receiving_probability` — fraction of realizations in which the
  cell was hit by an ember.

These let you separate fire spread driven by surface contagion from spread
driven by ember transport. See [Simulation Outputs](outputs.md) for the full
list of output fields.

## Example

A complete runnable scenario, including visualization of the generation and
receiving probability maps, is available in the repository at
`example/example_spotting_dynamics.py`.
