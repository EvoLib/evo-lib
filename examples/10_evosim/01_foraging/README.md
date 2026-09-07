# EvoSim Foraging

A persistent 2D world with food, energy, death, and asexual reproduction.

Each Forager contains an EvoLib `Indiv` with a `ParaComposite` holding an EvoNet
controller plus three sensor vectors: angles, fields of view, and ranges. The reference
configuration keeps all sensor modules fixed. Later experiments can enable mutation for
individual sensor properties without changing the simulation code.

The controller receives three local food-sensor activations and normalized energy. Each
food sensor has an angle, FOV, and range. It returns the strongest visible food signal as
`1 - distance / range`; FOV boundaries are hard, overlapping sensors are allowed, and
sensor geometry follows the toroidal world.

There are no generations and no explicit fitness function. Selection emerges from
survival and reproduction inside the shared world. The simulation stops after the
configured number of steps or when the population becomes extinct.

Run headless:

```bash
python examples/10_evosim/01_foraging/run.py
```

Render the same configured simulation:

```bash
python examples/10_evosim/01_foraging/run.py --render
```

`seed`, `steps`, and CSV metric logging are configured only in `simulation.yaml`.
Rendering does not change those simulation settings. In the Pygame view, press `S` to
toggle the sensor overlay and `ESC` to quit.
