import meep as mp
import numpy as np

freq = 2.4e9 / 3e8  # = 8.0
domain = 0.30
cell_size = domain * 1.2  # = 0.36
pml_thick = domain * 0.1  # = 0.03
antenna_r = 0.18

print(f'Cell size: {cell_size:.3f} m')
print(f'PML thickness: {pml_thick:.3f} m')  
print(f'Usable area: +/- {cell_size/2 - pml_thick:.3f} m')
print(f'Antenna radius: {antenna_r:.3f} m')
print(f'PROBLEM: antenna at {antenna_r:.3f} > PML edge at {cell_size/2 - pml_thick:.3f}!')
print('SOURCE IS INSIDE PML - gets absorbed!')
print()
print('FIX: Make cell bigger or antenna closer')

# Fix: cell = 0.60m, PML = 0.05m, usable = +/- 0.25m, antenna at 0.18 = OK
cell_fix = 0.60
pml_fix = 0.05
print(f'\nFIXED: cell={cell_fix}, PML={pml_fix}, usable=+/-{cell_fix/2-pml_fix:.2f}')
print(f'  Antenna at {antenna_r} < {cell_fix/2-pml_fix:.2f}? {antenna_r < cell_fix/2-pml_fix}')

# Quick test with fixed geometry
print('\n--- Running quick MEEP test with fixed geometry ---')
sources = [mp.Source(
    mp.ContinuousSource(frequency=freq),
    component=mp.Ez,
    center=mp.Vector3(antenna_r, 0),
)]

sim = mp.Simulation(
    cell_size=mp.Vector3(cell_fix, cell_fix),
    boundary_layers=[mp.PML(thickness=pml_fix)],
    geometry=[],  # empty - just air
    sources=sources,
    resolution=20,
    force_complex_fields=True,
)

sim.run(until=25.0)  # 200 periods at freq=8

rx_pos = mp.Vector3(-antenna_r, 0)  # RX opposite
ez = sim.get_field_point(mp.Ez, rx_pos)
print(f'\nRX field: |Ez| = {abs(ez):.6f}, phase = {np.angle(ez):.4f}')
print(f'SUCCESS!' if abs(ez) > 1e-10 else 'STILL ZERO - need more time')
