class Symmetry:
    def __init__(self, nSites):
        self.nSites = nSites
        self.perm = range(nSites)  # initialize to identity
        self.chi = complex(1.0)
        self.name = "id.N_" + str(self.nSites)

    def __repr__(self):
        return self.name + " | " + str(self.perm) + " | " + str(self.chi)

    def __mul__(self, otherSym):
        if otherSym.nSites is not self.nSites:
            raise Exception("Error: multiplied symetries have distinct sizes.")
        else:
            res = Symmetry(self.nSites)
            res.perm = [self.perm[otherSym.perm[s]] for s in range(self.nSites)]
            res.chi = self.chi * otherSym.chi
            res.name = self.name + "x" + otherSym.name
            return res

    def Apply(self, conf):
        '''conf is assumed to be a list or a tuple'''
        return [conf[i] for i in self.perm]

    def writeToFile(self, dirName):
        os.system("mkdir -p {0}".format(dirName))
        f = open(dirName + self.name, 'w')
        for el in self.perm:
            f.write(str(el) + ' ')
        f.write("(" + str(self.chi.real) + "," + str(self.chi.imag) + ")")
        f.close()
