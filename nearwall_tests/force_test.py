"""Momentum passed to the fluid by one particle near a z-wall (nearwall-fix check).

Box 4 x 4 x 6, AGRID = 1, stationary LB walls at z = 1 and z = 5 (halfway bounce-back).
A heavy particle at z = 1.03 moves with v = 0.1 through quiescent fluid, so the coupling
force gamma * v acts on the fluid. The fluid momentum is summed over fluid nodes only
(ESPResSo's own sum also counts boundary nodes). Stock 4.2.2 spreads ~47% of the force
onto the boundary node at z = 0.5, where it is lost; the patched build should pass ~100%.
"""
import numpy as np
import espressomd
import espressomd.lb
import espressomd.lbboundaries
import espressomd.shapes

AGRID, LZ, TAU, GAMMA, V, Z0 = 1.0, 6.0, 0.01, 1.0, 0.1, 1.03

system = espressomd.System(box_l=[4.0, 4.0, LZ])
system.time_step = TAU
system.cell_system.skin = 0.1
lbf = espressomd.lb.LBFluid(kT=0, agrid=AGRID, dens=1.0, visc=1.0 / 6.0 / TAU, tau=TAU)
system.actors.add(lbf)
system.thermostat.set_lb(LB_fluid=lbf, gamma=GAMMA, seed=1)
for normal, dist in (([0, 0, 1], AGRID), ([0, 0, -1], -(LZ - AGRID))):
    system.lbboundaries.add(espressomd.lbboundaries.LBBoundary(
        shape=espressomd.shapes.Wall(normal=normal, dist=dist), velocity=[0, 0, 0]))
p = system.part.add(pos=[2.0, 2.0, Z0], v=[V, 0, 0], mass=1e8)


def fluid_momentum_x():
    total = 0.0
    for i in range(4):
        for j in range(4):
            for k in range(1, int(LZ / AGRID) - 1):      # fluid nodes z = 1.5 .. 4.5
                node = lbf[i, j, k]
                total += node.density * node.velocity[0] * AGRID**3
    return total


print(f"espressomd from {espressomd.__file__}")
print(" steps  fluid_px   impulse_on_fluid  ratio")
p0 = p.mass * p.v[0]
for n in (1, 2, 5):
    system.integrator.run(n if n == 1 else n - prev)
    prev = n
    impulse = p0 - p.mass * p.v[0]                     # momentum the particle gave away
    px = fluid_momentum_x()
    print(f"{n:5d}  {px: .4e}  {impulse: .4e}  {px / impulse: .3f}")
