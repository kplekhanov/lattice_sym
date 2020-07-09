import os, cmath, sys

class Symmetry:
    def __init__(self, nSites):
        self.nSites = nSites
        self.perm = range(nSites) # initialize to identity
        self.chi = complex(1.0)
        self.name = "id.N_" + str(self.nSites)

    def __repr__(self):
        return self.name + " | " + str(self.perm) + " | " + str(self.chi)

    def __mul__(self, otherSym):
        if otherSym.nSites is not self.nSites:
            print "Error: multiplied symetries have distinct sizes."
            sys.exit(0)
        else:
            res = Symmetry(self.nSites)
            res.perm = [ self.perm[otherSym.perm[s]] for s in range(self.nSites) ]
            res.chi = self.chi * otherSym.chi
            res.name = self.name + "x" + otherSym.name
            return res

    def Apply(self, conf):
        "conf is assumed to be a list or tuple"
        return [ conf[i] for i in self.perm ]

    def writeToFile(self, dirName):
        os.system("mkdir -p {0}".format(dirName))
        f = open(dirName+self.name, 'w')
        for el in self.perm: f.write(str(el)+' ')
        f.write("("+str(self.chi.real)+","+str(self.chi.imag)+")")
        f.close()

# chain specialization
class ChainTranslation(Symmetry):
    def __init__(self, L, t=0, k=0):
        Symmetry.__init__(self, L)
        for s in range(L):
            res = (s+t+1)%L
            self.perm[s] = res-1 if res else L-1
        self.chi = cmath.exp(2*cmath.pi*t*k*1j/float(L))
        self.name = "trans.L_"+str(L)+".t_"+str(t)+".k_"+str(k)

class ChainReflection(Symmetry):
    def __init__(self, L, rev=0, r=1):
        Symmetry.__init__(self, L)
        if rev:
            for s in range(L):
                self.perm[s] = L-s-1
            self.chi = r
        else:
            self.chi = 1  # identity in this case
        self.name = "refl.L_"+str(L)+".rev_"+str(rev)+".r_"+str(r)

def createChainSymmetries(L, qn={ "k":0, "r":0 }):
    "Doesn't use reflection if r=0"
    k, r = qn["k"], qn["r"]
    if (k==0 or 2*abs(k)==L) and abs(r)==1:
        return [ ChainTranslation(L,t,k) * ChainReflection(L,rev,r) \
                 for rev in [0,1] for t in range(L) ]
    else: return [ ChainTranslation(L,t,k) for t in range(L) ]

# ladder specialization or square lattice Lx*Ly
class LadderTranslation(Symmetry):
    def __init__(self, Lx, Ly, tx=0, ty=0, kx=0, ky=0):
        Symmetry.__init__(self, Lx*Ly)
        nSites = Lx*Ly
        for y in range(Ly):
            for x in range(Lx):
                xp = (x+tx)%Lx
                yp = (y+ty)%Ly
                self.perm[x + Lx*y] = xp + Lx*yp
        phase = tx*kx / float(Lx) + ty*ky / float(Ly)
        self.chi = cmath.exp(2*cmath.pi*1j*phase)
        self.name = "trans.Lx_"+str(Lx)+".Ly_"+str(Ly) \
                        +".tx_"+str(tx)+".ty_"+str(ty) \
                        +".kx_"+str(kx)+".ky_"+str(ky)

def createLadderSymmetries(Lx, Ly, qn={ "kx":0, "ky":0, "r":0 }):
    "Doesn't use reflection if r=0"
    kx, ky, r = qn["kx"], qn["ky"], qn["r"]
    return [ LadderTranslation(Lx, Ly, tx, ty, kx, ky) \
             for ty in range(Ly) for tx in range(Lx) ]

