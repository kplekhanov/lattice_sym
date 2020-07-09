# -*- coding: utf-8 -*-
import numpy as np
import sys, math, cmath, os
from matplotlib import pyplot as plt
from matplotlib import lines as lines
from Symmetry import Symmetry

class Translation(list):
    def __init__(self, translationArr):
        list.__init__(self)
        self[:] = translationArr
        
    def __mul__(self, otherTranslation):
        if len(self) == len(otherTranslation):
            return [self[otherTranslation[i]] for i in range(len(self))]
        else: sys.exit("Error. Translation multiplied to the Translation of inequal size.")
        
    def Apply(self, positions):
        if len(self) == len(positions):
            return [self[positions[i]] for i in range(len(self))]
        else: sys.exit("Error. Translation applied to the list of inequal size.")

class LatticeSym(object):
    def __init__(self, nCell=4, basisVecs=[np.array([1.,0.]),np.array([0.,1.])], basisTrans=[[1,0,3,2],[2,3,0,1]]):
        self.nCell = nCell
        self.basisVecs = basisVecs
        self.basisTrans = [Translation(trans) for trans in basisTrans]
        self.useSymmetries = False
        self.needCoords = False
        self.spinInversion = 0
        self.reflEx = 0

    def getName(self):
        if self.name != None: return self.name
        else: return "oO." + str(self.nCell)    
    
    def buildBasisTiltH(self, lx, ly, vecs, tilt=0, name="nd"):
        self.name = name+"."+str(lx)+"x"+str(ly)+"th"+str(tilt)
        self.nCell = lx*ly
        self.basisVecs = vecs
        arrTrans0, arrTrans1 = [], []
        for y in range(ly):
            for x in range(lx):
                arrTrans0.append( y*lx+(x+1)%lx )
                if y < ly-1 : arrTrans1.append( (y+1)*lx+x )
                if y == ly-1: arrTrans1.append( (x-tilt)%lx )
        self.basisTrans = [Translation(arrTrans0), Translation(arrTrans1)]
                
    def buildBasisTiltV(self, lx, ly, vecs, tilt=0, name="nd"):
        self.name = name+"."+str(lx)+"x"+str(ly)+"tv"+str(tilt)
        self.nCell = lx*ly
        self.basisVecs = vecs
        arrTrans0, arrTrans1 = [], []
        for x in range(lx):
            for y in range(ly):
                arrTrans1.append( ((y+1)%ly)*lx+x )
                if x < lx-1 : arrTrans0.append( y*lx+(x+1) )
                if x == lx-1: arrTrans0.append( ((y-tilt)%ly)*lx ) 
        self.basisTrans = [Translation(arrTrans0), Translation(arrTrans1)]
        
    def buildBasis1D(self, l, vec1D):
        self.name ="ch."+str(l)
        self.nCell = l
        self.basisVecs = [vec1D, np.array([0.,1.])]
        arrTrans0, arrTrans1 = [], []
        for x in range(l):
            arrTrans1.append(x)
            arrTrans0.append((x+1)%l)
        self.basisTrans = [Translation(arrTrans0), Translation(arrTrans1)]  

    def buildMain(self, nSiteInCell=1, inCellPos=[np.array([0.,0.])]):
        self.buildBasisKVecs()
        self.buildBravaisLattice()
        self.buildQns()
        self.nSiteInCell = nSiteInCell
        self.inCellPos = inCellPos

    def buildBasisKVecs(self):
        # reciprocal lattice vectors
        v1x, v1y = self.basisVecs[0][0], self.basisVecs[0][1]
        v2x, v2y = self.basisVecs[1][0], self.basisVecs[1][1]
        if (v1x == 0 and v2x == 0) or (v1y == 0 and v2y == 0) or (v1x/v2x == v1y/v2y):
            sys.exit("Error. Lattice vectors do not correspond to vectors of 2D lattice.")
        else:
            mat = np.array([[v1x,v1y,0,0], [0,0,v2x,v2y], [0,0,v1x,v1y], [v2x,v2y,0,0]])
            k4Vec = np.linalg.solve(mat, np.array([2*math.pi,2*math.pi,0,0]))
            k1x, k1y, k2x, k2y = k4Vec[0], k4Vec[1], k4Vec[2], k4Vec[3]
            k1Norm, k2Norm = math.sqrt(k1x**2+k1y**2), math.sqrt(k2x**2+k2y**2)
        self.basisKVecs = [np.array([k1x, k1y]),np.array([k2x, k2y])]
        self.basisKVecsNorm = [np.array([k1x/k1Norm, k1y/k1Norm]),np.array([k2x/k2Norm, k2y/k2Norm])]
        self.basisKVecsToPlot = [np.array([1., k1y/k2y]),np.array([k2x/k1x, 1.])]

    def buildBravaisLattice(self, verbose=False):
        # all translations, positions and qnsMax (lx and ly)
        identity = range(self.nCell)
        self.translations = {(0,0): identity[:]}
        self.positions = [(0,0) for n in range(self.nCell)]
        self.qnsMax = [self.nCell, self.nCell]
        
        configN = identity[:]
        for n in range(1, self.nCell):
            configN = self.basisTrans[0].Apply(configN)
            if configN == identity: 
                self.qnsMax[0] = n
                break
            if not configN in self.translations.values(): # actually, this condition is not required
                self.translations[(n,0)] = configN
                self.positions[configN[0]] = (n,0)
        
        configM = identity[:]
        for m in range(1, self.nCell):
            configM = self.basisTrans[1].Apply(configM)
            if configM == identity:
                self.qnsMax[1] = m
                break
            if not configM in self.translations.values(): # and this one too
                self.translations[(0,m)] = configM
                self.positions[configM[0]] = (0,m)
            config = configM[:]
            for n in range(1, self.nCell):
                config = self.basisTrans[0].Apply(config)
                if config == identity: break
                if not config in self.translations.values(): # only this condition is important
                    self.translations[(n,m)] = config
                    self.positions[config[0]] = (n,m)
        if not len(self.translations) == self.nCell:
            sys.exit("Error. Incorrect number of lattice translations ({0})".format(len(self.translations)))
        self.lx = self.qnsMax[0]
        self.ly = self.nCell/self.qnsMax[0]
        firstTiltedConf = self.basisTrans[1].Apply(self.translations[(0,self.ly-1)])
        self.tilt = (self.lx-firstTiltedConf[0])%self.lx

    def buildQns(self, verbose=False):            
        # quantum numbers
        self.qns = [ (p,q) for p in range(self.qnsMax[0]) for q in range(self.qnsMax[1]) ]
        identity = range(self.nCell)
        
        configN = identity
        for n in range(1,self.nCell+1):
            configN = self.basisTrans[0].Apply(configN)
            if configN == identity:
                for p,q in self.qns:
                    num = p*n/float(self.qnsMax[0])
                    if not num.is_integer():
                        self.qns.remove((p,q))
        
        configM = identity
        for m in range(1,self.nCell+1):
            configM = self.basisTrans[1].Apply(configM)
            if configM == identity:
                for p,q in self.qns:
                    num = q*m/float(self.qnsMax[1])
                    if not num.is_integer():
                        self.qns.remove((p,q))
            config = configM[:]
            for n in range(1,self.nCell+1):
                config = self.basisTrans[0].Apply(config)
                if config == identity:
                    for p,q in self.qns:
                        num = p*n/float(self.qnsMax[0]) + q*m/float(self.qnsMax[1])
                        if not num.is_integer():
                            self.qns.remove((p,q))        
        if not len(self.qns) == self.nCell:
            sys.exit("Error. Incorrect ammount of quantum numbers ({0}).".format(len(self.qns)))
        self.qns = sorted(self.qns, key=lambda qns: qns[1])
                        
    def getChi(self, (p,q), (n,m)):
        return cmath.exp(1j*2*np.pi*( (p*n)/float(self.qnsMax[0]) + (q*m)/float(self.qnsMax[1])))

    def getRecSpacePositionCart(self, (p, q)):
        k1x, k1y = self.basisKVecs[0][0], self.basisKVecs[0][1]
        k2x, k2y = self.basisKVecs[1][0], self.basisKVecs[1][1]            
        kx = p*k1x/self.qnsMax[0] + q*k2x/self.qnsMax[1]
        ky = p*k1y/self.qnsMax[0] + q*k2y/self.qnsMax[1]
        return (kx, ky)

    def getPositionCart(self, n, m, siteIndex=0):
        #n,m = self.positions[cellIndex]
        x = self.basisVecs[0][0]*n + self.basisVecs[1][0]*m 
        y = self.basisVecs[0][1]*n + self.basisVecs[1][1]*m
        x += self.inCellPos[siteIndex][0]
        y += self.inCellPos[siteIndex][1]
        return (x,y)

    def getAbsIndex(self, cellIndex, siteIndex=0):
        return cellIndex*self.nSiteInCell + siteIndex
        
    def buildOneSymmetry(self, (p,q), (n,m), reflEx=0):
        sym = Symmetry(self.nCell*self.nSiteInCell)
        if abs(reflEx) == 0:
            sym.chi = self.getChi((p,q), (n,m))
            #sym.name = "trans.t1_{0}.t2_{1}.k1_{2}.k2_{3}".format(n,m,p,q)
            sym.name = "Trans.Lx"+str(self.lx)+".Ly"+str(self.lx) \
                       +".tx"+str(n)+".ty"+str(m) \
                       +".kx"+str(p)+".ky"+str(q)
            sym.perm = [ self.getAbsIndex(self.translations[(n,m)][i],j) for i in range(self.nCell) for j in range(self.nSiteInCell) ]
        elif abs(reflEx) == 1:
            sym.chi = self.getChi((p,q), (n,m))*reflEx
            sym.name = "Trans.Lx"+str(self.lx)+".Ly"+str(self.lx) \
                       +".tx"+str(n)+".ty"+str(m) \
                       +".kx"+str(p)+".ky"+str(q) \
                       +".reflEx"+str(reflEx)
            dictReflexion6x2 = {0:1, 1:0, 2:3, 3:2, 4:5, 5:4, 6:7, 7:6, 8:9, 9:8, 10:11, 11:10, 12:13,
                                13:12, 14:15, 15:14, 16:17, 17:16, 18:19, 19:18, 20:21, 21:20, 22:23, 23:22}
            dictReflexion4x4 = {0:23, 1:22, 2:17, 3:16, 4:19, 5:18, 6:21, 7:20, 8:9, 9:8, 10:11, 11:10, 12:13, 13:12, 14:15, 15:14,
                                16:3, 17:2, 18:5, 19:4, 20:7, 21:6, 22:1, 23:0, 24:25, 25:24, 26:27, 27:26, 28:29, 29:28, 30:31, 31:30}
            if self.lx == 6 and self.ly == 2 and self.tilt%self.lx == 5:
                sym.perm = [ dictReflexion6x2[self.getAbsIndex(self.translations[(n,m)][i],j)] for i in range(self.nCell) for j in range(self.nSiteInCell) ]
            elif self.lx == 4 and self.ly == 4 and self.tilt%self.lx == 2:
                sym.perm = [ dictReflexion4x4[self.getAbsIndex(self.translations[(n,m)][i],j)] for i in range(self.nCell) for j in range(self.nSiteInCell) ]                
            else:
                sys.exit('Cannot use the reflexion symmetry')
        return sym

    def buildSymmetries(self):
        self.symmetries = {}
        for p,q in self.qns:
            self.symmetries[(p,q)] = []
            # sorting is important for cpp
            if self.reflEx == 0:
                for n,m in sorted(self.translations.keys()):
                    self.symmetries[(p,q)].append( self.buildOneSymmetry((p,q),(n,m)) )
            elif abs(self.reflEx) == 1:
                for i in [0, self.reflEx]:
                  for n,m in sorted(self.translations.keys()):
                    self.symmetries[(p,q)].append( self.buildOneSymmetry((p,q),(n,m), reflEx=i) )                
                
    def printSymmetries(self):
        for p,q in sorted(self.qns):
            print "Qns = ({0},{1})".format(p,q)
            for sym in self.symmetries[(p,q)]:
                print sym        

    def makeCppInput(self, fileName, initDir = ".", dirName="SymFiles/"):
        text = "Lattice\n" \
               "{\n" \
               "   Nsites = " + str(self.nCell*self.nSiteInCell) + "\n" \
               "   Type = generic\n" \
               "   lCellX = " + str(self.lx) + "\n" \
               "   lCellY = " + str(self.ly) + "\n" \
               "   tilt = " + str(self.tilt) + "\n" \
               "   name = " + str(self.name) + "\n"
        if self.useSymmetries:
            allNames = ""
            for sym in self.symmetries[self.simSymSector]:
                sym.writeToFile(dirName)
                allNames += sym.name+","
            allNames = allNames[:-1]
            text += "   Symmetries\n" \
                    "   {\n" \
                    "     symmetry_sector = "
            for qn in self.simSymSector: text += str(qn) + " "
            text += "\n"
            text+="     symmetry_values = "
            for ki in self.getRecSpacePositionCart(self.simSymSector): text += str(ki) + " "
            text += "\n"
            text+= "     symmetries_directory = " + initDir + "/" + dirName + "\n" \
                   "     symmetries_files = " + allNames + "\n"
            if abs(self.spinInversion) == 1:
                text += "     usespinflip = yes\n" \
                        "     s = "+str(self.spinInversion)+"\n"
            text += "   }\n"
        if self.needCoords:
            text += "   CartCoords\n" \
                    "   {\n"
            textInd, textPos, textCartCoord = "", "", ""
            for i in range(self.nCell):
              for j in range(self.nSiteInCell):
                n, m = self.positions[i]
                textInd += str(i*self.nSiteInCell+j) + " "
                textPos += "("+str(n)+","+str(m)+","+str(j)+") "
                textCartCoord += str(self.getPositionCart(n,m,j)).replace(" ", "")+" "
            text += "     siteIndexArr = " + textInd + "\n"
            text += "     sitePosArr = " + textPos + "\n"
            text += "     siteCartCoordArr = " + textCartCoord + "\n"
            text += "   }\n"
        text += "}\n"
        fileName.write(text)
                        
    def plotLattice(self, fig=None, ax=None, s=150, save=False, show=False):
        if fig == None or ax == None: fig, ax = plt.subplots()
        x1, y1 = self.basisVecs[0][0], self.basisVecs[0][1]
        x2, y2 = self.basisVecs[1][0], self.basisVecs[1][1]
        # spanning vectors
        sx1, sy1 = self.basisVecs[0][0]*self.lx, self.basisVecs[0][1]*self.lx
        sx2, sy2 = self.basisVecs[1][0]*self.ly, self.basisVecs[1][1]*self.ly
        deltaX, deltaY = -x1/2.-x2/2., -y1/2.-y2/2.
        xArr = [deltaX,sx1+deltaX,sx1+sx2+deltaX,sx2+deltaX,deltaX]
        yArr = [deltaY,sy1+deltaY,sy1+sy2+deltaY,sy2+deltaY,deltaY]
        ax.plot(xArr, yArr, 'k', linewidth=2, zorder=10, color='black')
        #ax.plot(deltaX, deltaY, marker='x', color='green', ms=10., mew=2.)
        #ax.plot(sx1+deltaX, sy1+deltaY, marker='x', color='green', ms=10., mew=2.)
        #ax.plot(sx2+self.tilt*x1+deltaX, sy2+deltaY, marker='x', color='green', ms=10., mew=2.)
        # tilt
        ax.plot([sx1+deltaX,sx1+deltaX+x1/2.], [sy1+deltaY,sy1+deltaY+y1/2.], 'k', linewidth=2, zorder=10, color='black')
        ax.plot([sx1+sx2+deltaX,sx1+sx2+deltaX+x1/2.], [sy1+sy2+deltaY,sy1+sy2+deltaY+y1/2.], 'k', linewidth=2, zorder=10, color='black')
        ax.plot([deltaX,deltaX-x1/2.], [deltaY,deltaY-y1/2.], 'k', linewidth=2, zorder=10, color='black')
        ax.plot([sx2+deltaX,sx2+deltaX-x1/2.], [sy2+deltaY,sy2+deltaY-y1/2.], 'k', linewidth=2, zorder=10, color='black')
        if self.tilt == 0:
          ax.plot([sx1+deltaX,sx1+deltaX-x2/2.], [sy1+deltaY,sy1+deltaY-y2/2.], 'k', linewidth=2, zorder=10, color='black')
          ax.plot([sx1+sx2+deltaX,sx1+sx2+deltaX+x2/2.], [sy1+sy2+deltaY,sy1+sy2+deltaY+y2/2.], 'k', linewidth=2, zorder=10, color='black')
          ax.plot([deltaX,deltaX-x2/2.], [deltaY,deltaY-y2/2.], 'k', linewidth=2, zorder=10, color='black')
          ax.plot([sx2+deltaX,sx2+deltaX+x2/2.], [sy2+deltaY,sy2+deltaY+y2/2.], 'k', linewidth=2, zorder=10, color='black')
        else:
          ax.plot([sx1-self.tilt*x1+deltaX,sx1-self.tilt*x1+deltaX-x2/2.], [sy1+deltaY,sy1+deltaY-y2/2.], 'k', linewidth=2, zorder=10, color='black')
          ax.plot([sx2+self.tilt*x1+deltaX,sx2+self.tilt*x1+deltaX+x2/2.], [sy1+sy2+deltaY,sy1+sy2+deltaY+y2/2.], 'k', linewidth=2, zorder=10, color='black')
        # plotting all bravais lattice points in red and sites in blue
        # if nSiteInCel != 1 (for honeycomb lattice for example), plot all lattice points in blue
        dxTxt = (x1 + x2)*0.08
        dyTxt = (y1 + y2)*0.03
        xBasisArr, yBasisArr = [], []
        xAllArr, yAllArr = [], []
        for i in range(len(self.positions)):
            n, m = self.positions[i]
            x, y = n*x1+m*x2, n*y1+m*y2
            xBasisArr.append(x)
            yBasisArr.append(y)
            if self.nSiteInCell > 1:
                for j in range(self.nSiteInCell):
                    dx, dy = self.inCellPos[j][0], self.inCellPos[j][1]
                    xAllArr.append(x + dx)
                    yAllArr.append(y + dy)
                    ax.text(x+dx+dxTxt, y+dy+dyTxt, str(self.getAbsIndex(i,j)), fontsize=10, zorder=30, color='blue')
            else: ax.text(x+dxTxt*2, y+dyTxt*2, str(i)+"~("+str(n)+","+str(m)+")", fontsize=10, zorder=30, color='blue')
        ax.scatter(xBasisArr, yBasisArr, s/6, alpha=1.0, color='0.5', zorder=20)
        ax.scatter(xAllArr, yAllArr, s/3, alpha=1.0, color='red', zorder=20)
        plt.title("Cluster structure", fontsize=20)
        if show:
            plt.show()
        if save:
            os.system("mkdir -p outputs")
            os.system("mkdir -p outputs/" + self.name)
            os.system("mkdir -p outputs/" + self.name + "/fig")
            fig.savefig("outputs/" + self.name + "/fig/lattice.pdf")
        return fig, ax

    def plotBZ(self, fig=None, ax=None, s=200, showQns=True, reducedQns=True, save=False, show=False, extended=False):
        if fig == None or ax == None: fig, ax = plt.subplots()
        # plot points in the Brillouin zone in units of RL vectors -> in units of 2*pi
        k1x, k1y = self.basisKVecsToPlot[0][0], self.basisKVecsToPlot[0][1]
        k2x, k2y = self.basisKVecsToPlot[1][0], self.basisKVecsToPlot[1][1]
        # plot normalized reciprocal lattice vectors
        ax.plot([0,k1x,k1x+k2x,k2x,0], [0,k1y,k1y+k2y,k2y,0], 'k', linewidth=2, zorder=10)
        if extended:
            ax.plot([-k1x-k2x,k1x-k2x,k1x+k2x,k2x-k1x,-k1x-k2x], [-k1y-k2y,k1y-k2y,k1y+k2y,k2y-k1y,-k1y-k2y], 'k', linewidth=2, zorder=10)
            ax.plot([-k1x,k1x], [-k1y,k1y], 'k', linewidth=2, zorder=10)
            ax.plot([-k2x,k2x], [-k2y,k2y], 'k', linewidth=2, zorder=10)
        # obtained values in blue
        dkxTxt = (k1x/self.lx + k2x/self.ly)*0.12
        dkyTxt = (k1y/self.lx + k2y/self.ly)*0.12
        kxArr, kyArr = [], []
        for p, q in self.qns:
            kx = p*k1x/self.qnsMax[0] + q*k2x/self.qnsMax[1]
            ky = p*k1y/self.qnsMax[0] + q*k2y/self.qnsMax[1]
            kxArrTmp = [kx]
            kyArrTmp = [ky]
            if extended:
                kxArrTmp += [kx-k1x,kx-k2x,kx-k1x-k2x]
                kyArrTmp += [ky-k1y,ky-k2y,ky-k1y-k2y]
            kxArr += kxArrTmp
            kyArr += kyArrTmp
            if showQns:
              if reducedQns: text = str(self.qns.index((p,q)))
              else: text = "({0}/{1},{2}/{3})".format(p,self.qnsMax[0],q,self.qnsMax[1])
              for kxt, kyt in zip(kxArrTmp, kyArrTmp):
                ax.text(kxt+dkxTxt, kyt+dkyTxt, text, fontsize=12, color="blue", zorder=30)
                    
        ax.scatter(kxArr, kyArr, s, alpha=1.0, zorder=20, color="red")
        plt.subplots_adjust(left=0.17, bottom=0.16, right=0.95, top=None, wspace=None, hspace=None)
        #plt.title("Allowed quantum numbers", fontsize=20)
        plt.xticks(np.arange(0., 1.25, .25), fontsize = 20)
        plt.xlabel("$k_x$", fontsize=30, labelpad=-2)
        plt.yticks(np.arange(-.5, 1.25, .25), fontsize = 20)
        plt.ylabel("$k_y$", fontsize=30, labelpad=0)
        if extended:
            plt.xticks(np.arange(-1.0, 1.25, .5), fontsize = 20)
            plt.yticks(np.arange(-.5, 1.75, .5), fontsize = 20)
        #plt.pause(0.0001)
        if show:
            plt.show()
        if save:
            os.system("mkdir -p outputs")
            os.system("mkdir -p outputs/" + self.name)
            os.system("mkdir -p outputs/" + self.name + "/fig")
            fig.savefig("outputs/" + self.name + "/fig/qns.pdf")
        return fig, ax

























