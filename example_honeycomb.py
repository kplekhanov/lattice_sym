import numpy as np
import matplotlib.pyplot as plt
from latticeSym.LatticeSym import LatticeSym

lx, ly, tilt, nSiteInCell = 6, 2, -1, 2
vecs = [np.array((np.sqrt(3),0)), np.array((np.sqrt(3)/2,3/2))]

s = LatticeSym()
s.buildBasisTiltH(lx, ly, vecs, tilt, "sym.bkmHb.is")
s.buildMain(nSiteInCell, [np.array((0,0)), np.array((0,0.5))])
s.buildSymmetries()
print("Qns allowed by the cluster:", s.qns)

fig = plt.figure(figsize=(9,4))

fax1 = fig.add_axes((0.10, 0.20, 0.35, 0.70))
s.plotLattice(fig, fax1)

fax2 = fig.add_axes((0.575, 0.20, 0.35, 0.70))
s.plotBZ(fig, fax2)

plt.show()
