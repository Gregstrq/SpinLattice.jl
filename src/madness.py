import numpy as np
import numpy.linalg as nlinalg
from scipy import sparse
import matplotlib.pyplot as plt
import time
from itertools import product as cartesian_product
from operator import xor

def pauliX(vec, pos):
    mask = 1<<pos
    return 1, (vec ^ mask)
def pauliY(vec, pos):
    mask = 1<<pos
    if (vec & mask) == 0:
        sign = -1j
    else:
        sign = 1j
    return sign, (vec ^ mask)
def pauliZ(vec, pos):
    mask = 1<<pos
    if  (vec & mask) == 0:
        sign = -1
    else:
        sign = 1
    return sign, vec
pauli = [pauliX, pauliY, pauliZ]
paulid = {'x':pauliX, 'y':pauliY, 'z':pauliZ}

#########################################################################################
def changeBit(vector, spin):
        return vector ^ (1<<spin)
def invert(vector, spin1, spin2):
    return vector ^ ((1<<spin1) + (1<<spin2))
def getSign(vector, spin1, spin2):
    bit1 = (vector >> spin1) & 1
    bit2 = (vector >> spin2) & 1
    return 1 - ((bit1^bit2)<<1)

def cross3_3_3():
    q_spins = []
    for q_spin in np.ndindex((3,3,3)):
        if np.sum(np.array(q_spin)%(2,2,2))>1:
            q_spins.append(q_spin)
    return q_spins
def block(dims):
    return [index for index in np.ndindex(dims)]
def columnZ9():
    return [(0,0,i) for i in xrange(9)]
def diag9_011():
    return [(0,i,i) for i in xrange(9)]
def diag9_111():
    return [(i,i,i) for i in xrange(9)]

#########################################################################################

class UniformLattice(object):
    classPrefix = 'ul'
    def __init__(self, dims, hz = (0., 0., 0.), Js = (-0.41, -0.41, 0.82), Tmax = 40., tstep = 0.0078125, cb = True, pos_list = None):
        self.n = np.prod(dims)
        self.N = 2**self.n
        self.dims = np.array(dims)
        self.D = len(self.dims)
        self.init_indexTables()
        self.generate_Increments()
        self.hz = hz
        self.Jsinit = np.array(Js)
        self.Js = self.Jsinit/4.
        self.Tmax = Tmax
        self.tstep = tstep
        self.id6 = 1/6.
        self.init_Hamiltonian(cyclic_boundaries = cb)
        if pos_list is None:
            pos_list = xrange(self.n)
        self.init_Magnetization(pos_list)
        self.ts = np.arange(0, Tmax+tstep, tstep)
        self.initIntegrator()
        self.logfile = None
    def init_indexTables(self):
        self.dTable = np.arange(self.n).reshape(self.dims)
        self.rTable = np.ndarray((self.n,len(self.dims)),dtype=int)
        for index, val in np.ndenumerate(self.dTable):
           self.rTable[val] = np.array(index)
    def randomState(self, threshold = 2**(-7)):
        N = self.N
        amps = (1.0 - threshold)*np.random.rand(N) + threshold
        amps = np.sqrt(-np.log(amps)/self.N)
        amps = amps/np.linalg.norm(amps)
        phases = np.exp(2*np.pi*1j*np.random.rand(N))
        return amps*phases
    def generate_Increments(self):
        self.deltas = np.eye(self.D, dtype = int)
    def count_nonCB_links(self):
        counter = self.D*self.n
        for i in xrange(self.D):
            counter -= self.n/self.dims[i]
        return counter
    def init_Magnetization(self, pos_list):
        self.Ms = {}
        for i in [0,1,2]:
            row = np.ndarray((self.N*len(pos_list)),dtype=int)
            col = np.ndarray((self.N*len(pos_list)),dtype=int)
            if i==1:
                data = np.ndarray((self.N*len(pos_list)),dtype=complex)
            else:
                data = np.ndarray((self.N*len(pos_list)),dtype=float)
            index = 0
            for vec in xrange(self.N):
                for pos in pos_list:
                    data[index], row[index] = pauli[i](vec, pos)
                    col[index] = vec
                    index+=1
            self.Ms[i] = sparse.coo_matrix((data, (row, col)), shape=(self.N, self.N)).tocsr()
    def init_Hamiltonian(self, cyclic_boundaries = True):
        self.cb = cyclic_boundaries
        if self.hz[1] == 0.:
            dtype = np.float64
        else:
            dtype = np.complex128
        correction = (self.hz[0]!=0.)+(self.hz[1]!=0.)
        if self.cb:
            multiplier = self.n*self.D
        else:
            multiplier = self.count_nonCB_links()
        row = np.ndarray((self.N*(multiplier + self.n*correction + 1)), dtype = int)
        col = np.ndarray((self.N*(multiplier + self.n*correction + 1)), dtype = int)
        data = np.ndarray((self.N*(multiplier + self.n*correction + 1)), dtype = dtype)
        index = 0
        for vec in xrange(self.N):
            row[index], col[index], data[index] = vec, vec, 0
            for pos in xrange(self.n):
                val, vec2 = pauliZ(vec, pos)
                data[index] -= 0.5*self.hz[2]*val
            for pos in xrange(self.n):
                for delta_pos in self.deltas:
                    try:
                        sign = getSign(vec, pos, self.add_pos(pos, delta_pos))
                        data[index] -= sign*self.Js[2]
                    except IndexError:
                        pass
            if abs(data[index])<1e-9:
                data[index]=0.
            index += 1
            for pos in xrange(self.n):
                for delta_pos in self.deltas:
                    try:
                        pos2 = self.add_pos(pos, delta_pos)
                        vec2 = invert(vec, pos, pos2)
                        sign = getSign(vec, pos, pos2)
                        row[index], col[index], data[index] = vec2, vec, (sign*self.Js[1]-self.Js[0])
                        if abs(data[index])<1e-9:
                            data[index]=0.
                        index+=1
                    except IndexError:
                        pass
            if self.hz[0]!=0.:
                for pos in xrange(self.n):
                    val, row[index] = pauli[0](vec, pos)
                    col[index], data[index] = vec, -0.5*val*self.hz[0]
                    index += 1
            if self.hz[1]!=0.:
                for pos in xrange(self.n):
                    val, row[index] = pauli[1](vec, pos)
                    col[index], data[index] = vec, -0.5*val*self.hz[1]
                    index += 1
        self.H = sparse.csr_matrix((data, (row, col)), shape=(self.N, self.N))
    def add_pos(self, pos, delta_pos):
        if self.cb:
            return self.dTable[tuple((self.rTable[pos] + delta_pos) % self.dims)]
        else:
            return self.dTable[tuple(self.rTable[pos] + delta_pos)]
    def initState(self, threshold = 2**(-7)):
        self.psi = self.randomState(threshold = threshold)
        for corrType in xrange(3):
            self.psi_aid[corrType] = self.Ms[corrType].dot(self.psi)
        self.cors = np.zeros((3, len(self.ts)), dtype = float)
        self.i = 0
    def initIntegrator(self):
        self.newBlocks = np.ndarray(self.N, dtype = complex)
        self.k1 = np.ndarray(self.N, dtype = complex)
        self.k2 = np.ndarray(self.N, dtype = complex)
        self.k3 = np.ndarray(self.N, dtype = complex)
        self.k4 = np.ndarray(self.N, dtype = complex)
        self.psi = np.ndarray(self.N, dtype = complex)
        self.psi_aid = np.ndarray((3, len(self.psi)), dtype = complex)
    def setParameters(self, Tmax):
        self.Tmax = Tmax
        self.ts = np.arange(0, self.Tmax+self.tstep, self.tstep)
    def rhs(self, psi):
        self.newBlocks = self.H.dot(psi)
        self.newBlocks *= -1j*self.tstep
        return self.newBlocks
    def rungeKuttaStep(self, psi):
        tstep = self.tstep
        self.k1[:] = self.rhs(psi)
        self.k2[:] = self.rhs(psi + self.k1*0.5)
        self.k3[:] = self.rhs(psi + self.k2*0.5)
        self.k4[:] = self.rhs(psi + self.k3)
        psi += (self.k1 + 2*self.k2 + 2*self.k3 + self.k4)*self.id6
    def updateCorrelator(self):
        for corrType in xrange(3):
            cor = np.vdot(self.psi, self.Ms[corrType].dot(self.psi_aid[corrType]))
            self.cors[corrType, self.i] = cor
    def updateState(self):
        self.rungeKuttaStep(self.psi)
        for corrType in xrange(3):
            self.rungeKuttaStep(self.psi_aid[corrType])
        self.i += 1
    def output(self, var):
        if self.logfile is None:
            print var
        else:
            self.logfile.write(str(var)+'\n')
    def propagate(self):
        self.elapsedTime = -time.time()
        self.initState()
        t = 0.
        Tstop = self.ts[-1]
        self.updateCorrelator()
        while t  < Tstop:
            t += self.tstep
            self.updateState()
            self.updateCorrelator()
        self.cors = np.array(self.cors).real
        self.elapsedTime += time.time()
        self.output(self.elapsedTime)
    def generateFileName(self, postfix = 'txt', **kwargs):
        converter = lambda x: x if x is not None else ""
        nTrials = kwargs.get('nTrials')
        Tmax = self.Tmax
        sfile = "{prefix}n{n}H({Hx:.3f},{Hy:.3f},{Hz:.3f})J({Jx:.3f},{Jy:.3f},{Jz:.3f})Tmax{Tmax:.1f}nTr{nTrials:d}.{postfix}".format(prefix = self.classPrefix, n = self.n,\
                Hx = self.hz[0], Hy = self.hz[1], Hz = self.hz[2], Jx = 4*self.Js[0], Jy = 4*self.Js[1], Jz = 4*self.Js[2], Tmax = Tmax, nTrials = converter(nTrials), postfix = postfix)
        return sfile
    def averagePropagate(self, sfile = None, logfile = False, nTrials = 10):
        if logfile == True:
            logfile = self.generateFileName(nTrials = nTrials, postfix = 'log')
            self.logfile = open(logfile, 'w', 1)
        halffile = self.generateFileName(nTrials = nTrials, postfix = 'half')
        if sfile is None:
            sfile = self.generateFileName(nTrials = nTrials)
        self.totalTime = -time.time()
        self.propagate()
        self.corsx = self.cors.copy()
        np.savetxt(sfile, self.corsx)
        self.output(1)
        for i in xrange(nTrials - 1):
            self.propagate()
            self.corsx += self.cors
            self.output(i+2)
            np.savetxt(sfile, self.corsx)
            if i == nTrials/2:
                np.savetxt(halffile, self.corsx)
        self.cors = self.corsx
        self.totalTime += time.time()
        self.output(self.totalTime)
        if logfile == True:
            self.logfile.close()
            self.logfile = None
        np.savetxt(sfile, self.cors)
    def plotData(self):
        plt.ion()
        for cor in self.cors:
            corR = cor
            plt.plot(self.ts[:len(corR)], corR/corR[0])

#########################################################################################

class UniformLatticeL(UniformLattice):
    def propagate(self):
        self.elapsedTime = -time.time()
        self.initState()
        t = 0.
        Tstop = self.ts[-1]
        self.updateCorrelator()
        while t  < Tstop:
            t += self.tstep
            dTime = -time.time()
            self.updateState()
            self.updateCorrelator()
            dTime += time.time()
            self.output(t)
            self.output(dTime)
        self.cors = np.array(self.cors).real
        self.elapsedTime += time.time()
        self.output(self.elapsedTime)


#########################################################################################

class ClusteredLattice(UniformLattice):
    classPrefix = 'cll'
    def __init__(self, block_dims, blocks, hz = (0.,0.,0.), Js = (-0.41, -0.41, 0.82), Tmax = 10., tstep = 0.0078125, delimiter = 10, Modes = [('x', 'L'), ('y', 'L'), ('z', 'L')]):
        self.n = np.prod(block_dims)
        self.b = np.prod(blocks)
        self.N = 2**self.n
        self.dims = block_dims
        self.blocks = blocks
        self.D = len(self.dims)
        self.init_indexTables()
        self.generate_Increments()
        self.find_boundaryPos()
        self.construct_Links()
        self.hz = hz
        self.Jsinit = np.array(Js)
        self.Js = self.Jsinit/4.
        self.Tmax = Tmax
        self.tstep = tstep
        self.delimiter = delimiter
        self.id6 = 1/6.
        self.init_Hamiltonian(cyclic_boundaries = False)
        #self.init_Magnetization()
        self.init_SpinOperators()
        self.Js *= np.sqrt(self.N+1)
        self.ts = np.arange(0, delimiter*Tmax+tstep, tstep)
        self.initIntegrator()
        self.setModes(Modes)
        self.logfile = None
    def initIntegrator(self):
        self.psi = np.ndarray((self.b, self.N), dtype = complex)
        self.newPsi = np.ndarray((self.b, self.N), dtype = complex)
        self.k1 = np.ndarray((self.b, self.N), dtype = complex)
        self.k2 = np.ndarray((self.b, self.N), dtype = complex)
        self.k3 = np.ndarray((self.b, self.N), dtype = complex)
        self.k4 = np.ndarray((self.b, self.N), dtype = complex)
        self.Mvals = np.ndarray((self.b, len(self.edge_spins), 3), dtype = complex)
        self.MFcoefs = np.ndarray((self.b, len(self.edge_spins), 3), dtype = complex)
    def setParameters(self, Tmax = None, delimiter = None, Modes = None):
        if delimiter is not None:
            self.delimiter = delimiter
        if Tmax is not None:
            self.Tmax = Tmax
            self.ts = np.arange(0, self.delimiter*self.Tmax+self.tstep, self.tstep)
        if Modes is not None:
            self.setModes(Modes)
    def init_indexTables(self):
        super(ClusteredLattice, self).init_indexTables()
        self.BdTable = np.arange(self.b).reshape(self.blocks)
        self.BrTable = np.ndarray((self.b,len(self.blocks)),dtype=int)
        for index, val in np.ndenumerate(self.BdTable):
           self.BrTable[val] = np.array(index)
    def find_boundaryPos(self):
        self.edge_spins = []
        for pos in np.ndindex(tuple(self.dims)):
            for i in xrange(self.D):
                if pos[i]==0 or pos[i]==self.dims[i]-1:
                    self.edge_spins.append(self.dTable[pos])
                    break
        self.le = len(self.edge_spins)
    def construct_Links(self):
        self.links = []
        for block in xrange(self.b):
            for s in xrange(len(self.edge_spins)):
                for delta_pos in self.deltas:
                    for sign in [-1,1]:
                        newIndex = self.rTable[self.edge_spins[s]] + sign*delta_pos
                        newSpin = self.dTable[tuple(newIndex % self.dims)]
                        deltaBlock = newIndex/self.dims
                        #if block == 0:
                        #    print self.edge_spins[s]
                        #    print self.rTable[self.edge_spins[s]]
                        #    print sign*delta_pos
                        #    print newIndex % self.dims
                        #    print newSpin
                        #    print self.BrTable[block]
                        #    print deltaBlock
                        #    print (self.BrTable[block] + deltaBlock)%self.blocks
                        #    print "\n"
                        newBlock = self.BdTable[tuple((self.BrTable[block] + deltaBlock)%self.blocks)]
                        if newBlock != block:
                            self.links.append(((block, s),(newBlock, self.edge_spins.index(newSpin))))
                            #if block == 0:
                            #    print ((block, self.edge_spins[s]),(newBlock, newSpin))
    def init_SpinOperators(self):
        self.taus = []
        for pos in self.edge_spins:
            self.taus.append([])
            for i in range(3):
                row = np.ndarray(self.N, dtype=int)
                col = np.ndarray(self.N, dtype=int)
                data = np.ndarray(self.N, dtype=complex)
                for vec in xrange(self.N):
                    data[vec], row[vec] = pauli[i](vec, pos)
                    col[vec] = vec
                self.taus[-1].append(sparse.coo_matrix((data, (row, col)), shape=(self.N, self.N)).tocsr())
    def getMiddleSpins(self):
        ranges = []
        for i in xrange(self.D):
            if self.dims[i]%2 == 0:
                rangeList = [self.dims[i]/2-1, self.dims[i]/2]
            else:
                rangeList = [self.dims[i]/2]
            ranges.append(rangeList)
        posList = [self.dTable[e] for e in cartesian_product(*tuple(ranges))]
        return posList
    def buildOperator(self, op):
        assert isinstance(op, tuple)
        axis = op[0]
        otype = op[1]
        assert axis in ['x','y','z']
        assert otype in ['Global','Local','GlobalWithoutBorders']
        if otype == 'Global':
            posList = range(self.n)
        if otype == 'Local':
            posList = self.getMiddleSpins()
        if otype == 'GlobalWithoutBorders':
            posList = [pos for pos in range(self.n) if pos not in self.edge_spins]
        dtype = np.float64
        if axis == 'y':
            dtype = np.complex128
        ########################################
        size = self.N*len(posList)
        row = np.ndarray(size, dtype=int)
        col = np.ndarray(size, dtype=int)
        data = np.ndarray(size, dtype=dtype)
        index = 0
        for vec in xrange(self.N):
            for pos in posList:
                data[index], row[index] = paulid[axis](vec,pos)
                col[index] = vec
                index += 1
        self.ops[op] = sparse.coo_matrix((data, (row, col)), shape=(self.N, self.N)).tocsr()
    def initOperators(self):
        if not hasattr(self, 'ops'):
            self.ops = {}
        for op in self.operatorSet:
            if op not in self.ops:
                self.buildOperator(op)
    def initState(self, threshold = 2**(-10)):
        #self.psi = np.ndarray((self.b, self.N), dtype = complex)
        for i in xrange(self.b):
            self.psi[i] = self.randomState(threshold = threshold)
        self.means = {}
        for op in self.operatorSet:
            self.means[op] = []
        self.cors = []
        for i in xrange(len(self.Modes)):
            self.cors.append([])
    def updateState(self):
        self.rungeKuttaStep(self.psi)
    def updateCorrelator(self):
        for corrType in self.operatorSet:
            mean = 0.
            for i in xrange(self.b):
                mean += np.vdot(self.psi[i], self.ops[corrType].dot(self.psi[i]))
            self.means[corrType].append(mean)
    def setModes(self, Modes):
        self.modeList = {'G':('Global','Global'),'L':('Local','Global'),'GWB':('GlobalWithoutBorders', 'GlobalWithoutBorders'),'LGWB':('Local','GlobalWithoutBorders')}
        self.Modes = Modes
        self.operatorSet = set()
        for mode in Modes:
            for op in self.modeList[mode[1]]:
                self.operatorSet.add((mode[0], op))
        self.initOperators()
    def calculateMeanFieldCoefs(self, psi):
        #Mvals = np.ndarray((self.b, len(self.edge_spins), 3), dtype = complex)
        #MFcoefs = np.zeros((self.b, len(self.edge_spins), 3), dtype = complex)
        norms = np.ndarray(self.b, dtype = complex)
        for block in xrange(self.b):
            norms[block] = np.vdot(psi[block], psi[block])
        for block in xrange(self.b):
            for spin in xrange(self.le):
                for sigma in xrange(3):
                    self.Mvals[block, spin, sigma] = np.vdot(psi[block], self.taus[spin][sigma].dot(psi[block]))/norms[block]*self.Js[sigma]
        self.MFcoefs.fill(0)
        #print nlinalg.norm(self.MFcoefs)
        for link in self.links:
            self.MFcoefs[link[1]] -= self.Mvals[link[0]]
        #return MFcoefs
    def rhs(self, psi):
        self.calculateMeanFieldCoefs(psi)
        #self.newPsi = np.ndarray((self.b, self.N), dtype = complex)
        print np.vdot(psi[0],psi[0])
        for block in xrange(self.b):
            self.newPsi[block] = self.H.dot(psi[block])
            for sigma in xrange(3):
                for spin in xrange(self.le):
                    self.newPsi[block] += self.MFcoefs[block, spin, sigma]*self.taus[spin][sigma].dot(psi[block])
        self.newPsi *= -1j*self.tstep
        return self.newPsi
    def buildCors(self):
        start = 0
        for key in self.means.keys():
            self.means[key] = np.array(self.means[key])
        for i in xrange(len(self.Modes)):
            op1, op2 = self.modeList[self.Modes[i][1]]
            means1 = self.means[(self.Modes[i][0],op1)].copy()
            means2 = self.means[(self.Modes[i][0],op2)].copy()
            #print (self.Modes[i][0],op1)
            #print (self.Modes[i][0],op2)
            delta = 0
            stop = len(means1)
            maxDelta = (stop - start)/self.delimiter
            while delta <= maxDelta:
                #print i, delta
                val = np.sum(means1[start:stop-1-delta]*means2[start+delta:stop-1])
                val += np.sum(means2[start:stop-1-delta]*means1[start+delta:stop-1])
                self.cors[i].append(val*0.5/float(stop-start-delta))
                #print len(self.cors[i])
                delta += 1
        self.cors = np.array(self.cors)
    def propagate(self, psi = None):
        self.elapsedTime = -time.time()
        self.initState()
        if psi is not None:
            self.psi = psi.copy()
        t = 0.
        Tstop = self.ts[-1]
        self.updateCorrelator()
        while t  < Tstop:
            t += self.tstep
            self.updateState()
            self.updateCorrelator()
        self.buildCors()
        self.elapsedTime += time.time()
        self.output(self.elapsedTime)
    def generateFileName(self, postfix = 'txt', **kwargs):
        converter = lambda x: x if x is not None else ""
        nTrials = kwargs.get('nTrials')
        Tmax = self.Tmax
        delimiter = self.delimiter
        modes = ""
        for mode in self.Modes:
            modes += mode[0]+mode[1]+','
        modes = modes[:-1]
        sfile = "{prefix}n{n:d}d{nd}b{nb}H({Hx:.3f},{Hy:.3f},{Hz:.3f})J({Jx:.3f},{Jy:.3f},{Jz:.3f})Tmax{Tmax:.1f}Del{delimiter:d}nTr{nTrials:d}M{modes}.{postfix}".format(prefix = self.classPrefix, n = self.n*self.b, nd = self.dims, nb = self.blocks,\
                Hx = self.hz[0], Hy = self.hz[1], Hz = self.hz[2], Jx = self.Jsinit[0], Jy = self.Jsinit[1], Jz = self.Jsinit[2], Tmax = Tmax, delimiter = converter(delimiter), nTrials = converter(nTrials), modes = modes, postfix = postfix)
        return sfile


#########################################################################################


class HybridLattice(ClusteredLattice):
    classPrefix = 'hl'
    def __init__(self, block_dims, full_dims, hz = (0.,0.,0.), Js = (-0.41,-0.41,0.82), Tmax = 10., tstep = 0.0078125, delimiter = 10, Axes = [0,1,2]):
        self.n = np.prod(block_dims)
        self.nf = np.prod(full_dims)
        self.b = self.nf - self.n
        for i in xrange(len(full_dims)):
            assert full_dims[i]>block_dims[i]
        self.N = 2**self.n
        self.dims = block_dims
        self.full_dims = full_dims
        self.D = len(self.dims)
        self.init_indexTables()
        self.generate_Increments()
        self.find_boundaryPos()
        self.construct_Links()
        self.hz = hz
        self.Jsinit = np.array(Js)
        self.Js = self.Jsinit/4.
        self.Tmax = Tmax
        self.tstep = tstep
        self.delimiter = delimiter
        self.id6 = 1/6.
        self.Nroot = np.sqrt(self.N+1)
        self.iNroot = 1/self.Nroot
        self.ts = np.arange(0, delimiter*Tmax+tstep, tstep)
        self.init_Hamiltonian(cyclic_boundaries = False)
        self.init_SpinOperators()
        self.initIntegrator()
        self.setAxes(Axes)
        self.logfile = None
    def setParameters(self, Tmax = None, delimiter = None, Axes = None):
        if delimiter is not None:
            self.delimiter = delimiter
        if Tmax is not None:
            self.Tmax = Tmax
            self.ts = np.arange(0, self.delimiter*self.Tmax+self.tstep, self.tstep)
        if Axes is not None:
            self.setAxes(Axes)
    def randomClassicalState(self):
        state = np.ndarray((3, self.b), dtype = float)
        zs = 2*np.random.rand(self.b) - 1
        rs = np.sqrt(1-np.square(zs))
        angles = np.random.rand(self.b)*2*np.pi
        state[0] = rs*np.cos(angles)
        state[1] = rs*np.sin(angles)
        state[2] = zs
        state *= np.sqrt(3)/2.
        #state.transpose()
        return state
    def init_indexTables(self):
        super(ClusteredLattice, self).init_indexTables()
        self.FdTable = np.arange(self.nf).reshape(self.full_dims)
        self.FrTable = np.ndarray((self.nf,len(self.full_dims)),dtype=int)
        for index, val in np.ndenumerate(self.FdTable):
           self.FrTable[val] = np.array(index)
        self.cTable = range(self.nf)
        for index in np.ndindex(self.dims):
            self.cTable.remove(self.FdTable[index])
    def initIntegrator(self):
        self.psi = np.ndarray(self.N, dtype = complex)
        self.newPsi = np.ndarray(self.N, dtype = complex)
        self.clState = np.ndarray((3, self.b), dtype = float)
        self.Hfield = np.ndarray((3, self.b), dtype = float)
        self.newClState = np.ndarray((3, self.b), dtype = float)
        self.kq1 = np.ndarray(self.N, dtype = complex)
        self.kq2 = np.ndarray(self.N, dtype = complex)
        self.kq3 = np.ndarray(self.N, dtype = complex)
        self.kq4 = np.ndarray(self.N, dtype = complex)
        self.kcl1 = np.ndarray((3, self.b), dtype = float)
        self.kcl2 = np.ndarray((3, self.b), dtype = float)
        self.kcl3 = np.ndarray((3, self.b), dtype = float)
        self.kcl4 = np.ndarray((3, self.b), dtype = float)
        self.kappaCl = np.ndarray((3, self.le), dtype = float)
        self.kappaQ = np.ndarray((3, self.le), dtype = float)
    def buildOperator(self, op):
        assert isinstance(op, tuple)
        axis = op[0]
        otype = op[1]
        assert axis in [0,1,2]
        assert otype in ['Global','Local']
        if otype == 'Global':
            posList = range(self.n)
        if otype == 'Local':
            posList = self.getMiddleSpins()
        dtype = np.float64
        if axis == 1:
            dtype = np.complex128
        ########################################
        size = self.N*len(posList)
        row = np.ndarray(size, dtype=int)
        col = np.ndarray(size, dtype=int)
        data = np.ndarray(size, dtype=dtype)
        index = 0
        for vec in xrange(self.N):
            for pos in posList:
                data[index], row[index] = pauli[axis](vec,pos)
                col[index] = vec
                index += 1
        self.ops[op] = sparse.coo_matrix((data, (row, col)), shape=(self.N, self.N)).tocsr()
    def initOperators(self):
        if not hasattr(self, 'ops'):
            self.ops = {}
        for axis in self.axes:
            op = (axis, 'Local')
            if op not in self.ops:
                self.buildOperator(op)
            op = (axis, 'Global')
            if op not in self.ops:
                self.buildOperator(op)
    def construct_Links(self):
        self.FM = np.zeros((self.b,self.b), dtype = int)
        self.Q2CL = np.zeros((self.b, self.le), dtype = int)
        for i in xrange(self.b):
            pos = self.cTable[i]
            for delta in self.deltas:
                for sign in [-1,1]:
                    newIndex = (self.FrTable[pos] + sign*delta) % self.full_dims
                    newPos = self.FdTable[tuple(newIndex)]
                    #print pos
                    #print self.FrTable[pos]
                    #print sign*delta
                    #print newIndex
                    #print newPos
                    if newPos in self.cTable:
                        self.FM[i, self.cTable.index(newPos)] += 1
                        #print "\n"
                    else:
                        #print self.dTable[tuple(newIndex)]
                        edge_spin = self.edge_spins.index(self.dTable[tuple(newIndex)])
                        self.Q2CL[i, edge_spin] += 1
        self.FM = sparse.csr_matrix(self.FM)
        self.CL2Q = sparse.csr_matrix(self.Q2CL.T)
        self.Q2CL = sparse.csr_matrix(self.Q2CL)
    def setAxes(self, Axes):
        self.axes = Axes
        self.initOperators()
    def initOperators(self):
        if not hasattr(self, 'ops'):
            self.ops = {}
        for axis in self.axes:
            op = (axis, 'Local')
            if op not in self.ops:
                self.buildOperator(op)
            op = (axis, 'Global')
            if op not in self.ops:
                self.buildOperator(op)
    def initState(self, threshold = 2**(-10)):
        self.psi = self.randomState(threshold = threshold)
        self.clState = self.randomClassicalState()
        self.means = {}
        for op in self.ops.keys():
            self.means[op] = []
        self.cors = []
        for i in xrange(len(self.axes)):
            self.cors.append([])
    def updateState(self):
        self.rungeKuttaStep(self.psi, self.clState)
    def updateCorrelator(self):
        for axis in self.axes:
            self.means[(axis, 'Local')].append(np.vdot(self.psi, self.ops[(axis, 'Local')].dot(self.psi)))
            temp = np.vdot(self.psi, self.ops[(axis, 'Global')].dot(self.psi))
            self.means[(axis, 'Global')].append(temp + np.sum(self.clState[axis])*2*self.iNroot)
    def buildCors(self):
        start = 0
        for key in self.means.keys():
            self.means[key] = np.array(self.means[key])
        for i in xrange(len(self.axes)):
            means1 = self.means[(self.axes[i], 'Local')]
            means2 = self.means[(self.axes[i], 'Global')]
            delta = 0
            stop = len(means1)
            maxDelta = (stop - start)/self.delimiter
            while delta <= maxDelta:
                val = np.sum(means1[start:stop-1-delta]*means2[start+delta:stop-1])
                val += np.sum(means2[start:stop-1-delta]*means1[start+delta:stop-1])
                self.cors[i].append(val*0.5/float(stop-start-delta))
                delta += 1
        self.cors = np.array(self.cors)
    def calculateMeanFieldCoefs(self, psi, clState):
        norm = np.vdot(psi, psi)
        for sigma in xrange(3):
            for spin in xrange(self.le):
                self.kappaCl[sigma, spin] = np.vdot(psi, self.taus[spin][sigma].dot(psi))*self.Jsinit[sigma]*0.5*self.Nroot/norm
                self.kappaQ[sigma] = self.CL2Q.dot(clState[sigma])*self.Jsinit[sigma]*0.5
    def rhs(self, psi, clState):
        self.calculateMeanFieldCoefs(psi, clState)
        self.newPsi = self.H.dot(psi)
        for sigma in xrange(3):
            for spin in xrange(self.le):
                self.newPsi -= self.kappaQ[sigma, spin]*self.taus[spin][sigma].dot(psi)
            self.Hfield[sigma] = self.FM.dot(clState[sigma])*self.Jsinit[sigma]
            self.Hfield[sigma] += self.Q2CL.dot(self.kappaCl[sigma])
            self.Hfield[sigma] += self.hz[sigma]
        self.newPsi *= -1j*self.tstep
        self.newClState[0] = clState[1]*self.Hfield[2] - clState[2]*self.Hfield[1]
        self.newClState[1] = clState[2]*self.Hfield[0] - clState[0]*self.Hfield[2]
        self.newClState[2] = clState[0]*self.Hfield[1] - clState[1]*self.Hfield[0]
        self.newClState *= self.tstep
        return self.newPsi, self.newClState
    def rungeKuttaStep(self, psi, clState):
        self.kq1[:], self.kcl1[:] = self.rhs(psi, clState)
        self.kq2[:], self.kcl2[:] = self.rhs(psi + self.kq1*0.5, clState + self.kcl1*0.5)
        self.kq3[:], self.kcl3[:] = self.rhs(psi + self.kq2*0.5, clState + self.kcl2*0.5)
        self.kq4[:], self.kcl4[:] = self.rhs(psi + self.kq3, clState + self.kcl3)
        psi += (self.kq1 + 2*self.kq2 + 2*self.kq3 + self.kq4)*self.id6
        clState += (self.kcl1 + 2*self.kcl2 + 2*self.kcl3 + self.kcl4)*self.id6
    def propagate(self, psi = None, clState = None):
        self.elapsedTime = -time.time()
        self.initState()
        if psi is not None:
            self.psi = psi.copy()
        if clState is not None:
            self.clState = clState.copy()
        t = 0.
        Tstop = self.ts[-1]
        self.updateCorrelator()
        while t  < Tstop:
            t += self.tstep
            self.updateState()
            self.updateCorrelator()
        self.buildCors()
        self.elapsedTime += time.time()
        self.output(self.elapsedTime)
    def generateFileName(self, postfix = 'txt', **kwargs):
        converter = lambda x: x if x is not None else ""
        nTrials = kwargs.get('nTrials')
        Tmax = self.Tmax
        delimiter = self.delimiter
        axs = ""
        for axis in self.axes:
           axs  += str(axis) +','
        axs = axs[:-1]
        sfile = "{prefix}d{dims}fd{full_dims}H({Hx:.3f},{Hy:.3f},{Hz:.3f})J({Jx:.3f},{Jy:.3f},{Jz:.3f})Tmax{Tmax:.1f}Del{delimiter:d}nTr{nTrials:d}A{axes}.{postfix}".format(prefix = self.classPrefix, dims = self.dims, full_dims = self.full_dims, \
                Hx = self.hz[0], Hy = self.hz[1], Hz = self.hz[2], Jx = self.Jsinit[0], Jy = self.Jsinit[1], Jz = self.Jsinit[2], Tmax = Tmax, delimiter = converter(delimiter), nTrials = converter(nTrials), axes = axs, postfix = postfix)
        return sfile



#########################################################

class HybridLatticeIR(HybridLattice):
    classPrefix = 'hlIR'
    def __init__(self, q_spins, full_dims, middle_spins, hz = (0.,0.,0.), Js = (-0.41,-0.41,0.82), Tmax = 10., tstep = 0.0078125, delimiter = 10, Axes = [0,1,2], radius = None):
        self.middle_spins = middle_spins
        self.n = len(q_spins)
        self.nf = np.prod(full_dims)
        self.b = self.nf - self.n
        self.N = 2**self.n
        self.q_spins = q_spins
        self.full_dims = full_dims
        self.D = len(self.full_dims)
        self.init_indexTables()
        self.generate_Increments()
        self.find_boundaryPos()
        self.construct_Links()
        self.hz = hz
        self.Jsinit = np.array(Js)
        self.Js = self.Jsinit/4.
        self.Tmax = Tmax
        self.tstep = tstep
        self.delimiter = delimiter
        self.radius = radius
        self.id6 = 1/6.
        self.Nroot = np.sqrt(self.N+1)
        self.iNroot = 1/self.Nroot
        self.ts = np.arange(0, delimiter*Tmax+tstep, tstep)
        self.get_spins_inside_radius()
        self.init_Hamiltonian()
        self.init_SpinOperators()
        self.initIntegrator()
        self.setAxes(Axes)
        self.logfile = None
    def init_indexTables(self):
        self.dTable = {}
        for i in xrange(self.n):
            self.dTable[self.q_spins[i]] = i
        self.rTable = np.ndarray((self.n, len(self.full_dims)), dtype = int)
        for index in self.dTable.keys():
            self.rTable[self.dTable[index]] = np.array(index)
        self.FdTable = np.arange(self.nf).reshape(self.full_dims)
        self.FrTable = np.ndarray((self.nf,len(self.full_dims)),dtype=int)
        for index, val in np.ndenumerate(self.FdTable):
           self.FrTable[val] = np.array(index)
        self.cTable = range(self.nf)
        for index in self.dTable.keys():
            self.cTable.remove(self.FdTable[index])
    def getMiddleSpins(self):
        posList = [self.dTable[mid_spin] for mid_spin in self.middle_spins]
        return posList
    def find_boundaryPos(self):
        self.edge_spins = []
        self.q_links = []
        for index in xrange(self.n):
            status = 0
            pos = self.rTable[index]
            for delta_pos in self.deltas:
                for sigma in [-1,1]:
                    newPos_t = tuple((pos + sigma*delta_pos) % self.full_dims)
                    if newPos_t in self.dTable.keys():
                        if (self.dTable[newPos_t], index) not in self.q_links:
                            self.q_links.append((index, self.dTable[newPos_t]))
                    else:
                        status = 1
            if status == 1:
                self.edge_spins.append(index)
        self.le = len(self.edge_spins)
    def get_spins_inside_radius(self):
        if self.radius is None:
            self.CIR = range(self.b)
        else:
            self.CIR = []
            for cl_spin in self.cTable:
                dist = None
                for middle_spin in self.middle_spins:
                    pos1 = self.FrTable[cl_spin]
                    pos2 = np.array(middle_spin)
                    temp_dist = self.get_smallest_distance(pos1, pos2)
                    if temp_dist < dist or dist is None:
                        dist = temp_dist
                if dist<self.radius:
                    self.CIR.append(self.cTable.index(cl_spin))
                    self.CIR.sort()
    def get_smallest_distance(self,pos1, pos2):
        vec = pos2-pos1
        for i in xrange(self.D):
            if abs(vec[i]) > self.full_dims[i]/2.:
                sign = np.sign(vec[i])
                vec[i] -= sign*self.full_dims[i]
        return np.linalg.norm(vec)
    def updateCorrelator(self):
        for axis in self.axes:
            self.means[(axis, 'Local')].append(np.vdot(self.psi, self.ops[(axis, 'Local')].dot(self.psi)))
            temp = np.vdot(self.psi, self.ops[(axis, 'Global')].dot(self.psi))
            self.means[(axis, 'Global')].append(temp + np.sum(self.clState[axis][self.CIR])*2*self.iNroot)
    def init_Hamiltonian(self):
        if self.hz[1] == 0.:
            dtype = np.float64
        else:
            dtype = np.complex128
        correction = (self.hz[0]!=0.)+(self.hz[1]!=0.)
        row = np.ndarray((self.N*(len(self.q_links) + self.n*correction + 1)), dtype = int)
        col = np.ndarray((self.N*(len(self.q_links) + self.n*correction + 1)), dtype = int)
        data = np.ndarray((self.N*(len(self.q_links) + self.n*correction + 1)), dtype = dtype)
        index = 0
        for vec in xrange(self.N):
            row[index], col[index], data[index] = vec, vec, 0
            for pos in xrange(self.n):
                val, vec2 = pauliZ(vec, pos)
                data[index] -= 0.5*self.hz[2]*val
            for link in self.q_links:
                sign = getSign(vec, link[0], link[1])
                data[index] -= sign*self.Js[2]
            if abs(data[index])<1e-9:
                data[index]=0.
            index += 1
            for link in self.q_links:
                vec2 = invert(vec, link[0], link[1])
                sign = getSign(vec, link[0], link[1])
                row[index], col[index], data[index] = vec2, vec, (sign*self.Js[1]-self.Js[0])
                if abs(data[index])<1e-9:
                    data[index]=0.
                index+=1
            if self.hz[0]!=0.:
                for pos in xrange(self.n):
                    val, row[index] = pauli[0](vec, pos)
                    col[index], data[index] = vec, -0.5*val*self.hz[0]
                    index += 1
            if self.hz[1]!=0.:
                for pos in xrange(self.n):
                    val, row[index] = pauli[1](vec, pos)
                    col[index], data[index] = vec, -0.5*val*self.hz[1]
                    index += 1
        self.H = sparse.csr_matrix((data, (row, col)), shape=(self.N, self.N))
    def generateFileName(self, postfix = 'txt', **kwargs):
        converter = lambda x: x if x is not None else ""
        nTrials = kwargs.get('nTrials')
        Tmax = self.Tmax
        if self.radius is None:
            rad = ""
        else:
            rad = "r{r}".format(r=self.radius)
        delimiter = self.delimiter
        axs = ""
        for axis in self.axes:
           axs  += str(axis) +','
        axs = axs[:-1]
        sfile = "{prefix}n{n:d}fd{full_dims}{rad}H({Hx:.3f},{Hy:.3f},{Hz:.3f})J({Jx:.3f},{Jy:.3f},{Jz:.3f})Tmax{Tmax:.1f}Del{delimiter:d}nTr{nTrials:d}A{axes}.{postfix}".format(prefix = self.classPrefix, n = self.n, full_dims = self.full_dims, rad = rad,\
                Hx = self.hz[0], Hy = self.hz[1], Hz = self.hz[2], Jx = self.Jsinit[0], Jy = self.Jsinit[1], Jz = self.Jsinit[2], Tmax = Tmax, delimiter = converter(delimiter), nTrials = converter(nTrials), axes = axs, postfix = postfix)
        return sfile


#########################################################

class FIDLattice(HybridLatticeIR):
    classPrefix = "FIDL"
    def __init__(self, D, lattice_dims, q_spins, middle_spins, coupling = 1., d = 1, cell_vectors = [], basis_vectors = None, hz = (0.,0.,0.), Tmax = 10., tstep = 0.0078125, delimiter = 10, direction = (1,0,0)):
        self.D = D
        self.d = d
        self.coupling = coupling
        assert isinstance(lattice_dims, tuple)
        assert D == len(lattice_dims)# == len(q_spins[0]) == len(middle_spins[0])
        self.lattice_dims = lattice_dims
        self.full_dims = self.lattice_dims + (d,)
        self.cell_vectors = cell_vectors
        self.cell_vectors.append(tuple(np.zeros(D)))
        self.cell_vectors = np.array(self.cell_vectors)
        if basis_vectors is None:
            self.basis_vectors = np.eye(D)
        else:
            self.basis_vectors = np.array(basis_vectors)
        self.middle_spins = middle_spins
        self.q_spins = q_spins
        if d == 1:
            for i in xrange(len(self.middle_spins)):
                self.middle_spins[i] += (0,)
            for i in xrange(len(self.q_spins)):
                self.q_spins[i] += (0,)
        self.n = len(q_spins)
        self.N = 2**self.n
        self.nf = np.prod(self.full_dims)
        self.b = self.nf - self.n
        self.hz = hz
        self.direction = np.array(direction)
        self.ndirection = self.direction/nlinalg.norm(self.direction)
        self.Tmax = Tmax
        self.tstep = tstep
        self.delimiter = delimiter
        self.id6 = 1/6.
        self.id6 = 1/6.
        self.Nroot = np.sqrt(self.N+1)
        self.iNroot = 1/self.Nroot
        self.ts = np.arange(0, delimiter*Tmax+tstep, tstep)
        self.coefs = [-0.5,-0.5,1.]
        ###
        self.init_indexTables()
        self.generate_Increments()
        self.construct_Links()
        self.init_Hamiltonian()
        self.init_SpinOperators()
        self.initIntegrator()
        self.init_Operators()
        self.logfile = None
    def initIntegrator(self):
        self.psi = np.ndarray(self.N, dtype = complex)
        self.newPsi = np.ndarray(self.N, dtype = complex)
        self.clState = np.ndarray((3, self.b), dtype = float)
        self.Hfield = np.ndarray((3, self.b), dtype = float)
        self.newClState = np.ndarray((3, self.b), dtype = float)
        self.kq1 = np.ndarray(self.N, dtype = complex)
        self.kq2 = np.ndarray(self.N, dtype = complex)
        self.kq3 = np.ndarray(self.N, dtype = complex)
        self.kq4 = np.ndarray(self.N, dtype = complex)
        self.kcl1 = np.ndarray((3, self.b), dtype = float)
        self.kcl2 = np.ndarray((3, self.b), dtype = float)
        self.kcl3 = np.ndarray((3, self.b), dtype = float)
        self.kcl4 = np.ndarray((3, self.b), dtype = float)
        self.q_means = np.ndarray((3, self.n), dtype = float)
        self.q_field = np.ndarray((3, self.n), dtype = float)
    def generate_Increments(self):
        basis_increments = np.array([e for e in cartesian_product(*tuple([[-1,0,1]]*self.D))])
        bas_vec = self.basis_vectors.copy()
        for i in xrange(self.D):
            bas_vec[i] *= self.lattice_dims[i]
        self.deltas = basis_increments.dot(bas_vec)
    def get_vector(self, pos):
        vec = np.zeros((self.D,), dtype = float)
        for i in xrange(self.D):
            vec += pos[i]*self.basis_vectors[i]
        vec += self.cell_vectors[pos[-1]]
        return vec
    def get_smallest_vector(self, pos1, pos2):
        vec = self.get_vector(pos2) - self.get_vector(pos1)
        smallest = vec.copy()
        radius = nlinalg.norm(smallest)
        for delta in self.deltas:
            temp_vec = vec + delta
            temp_radius = nlinalg.norm(temp_vec)
            if temp_radius < radius:
                smallest = temp_vec.copy()
                radius = temp_radius
        #print pos1, pos2, smallest
        test_vec = np.ndarray((self.D + 1,), dtype = float)
        signs = np.sign(vec)
        for i in xrange(self.D):
            if abs(vec[i]) <= self.lattice_dims[i]/2.:
                signs[i] = 0
        test_vec = vec - signs*self.lattice_dims
        if not np.array_equal(test_vec, smallest):
            print pos1, pos2, smallest, test_vec
        if len(smallest)<len(self.ndirection):
            sm = smallest.copy()
            sm = np.append(sm, np.zeros(len(self.ndirection)-len(smallest)))
        else:
            sm = smallest
        return radius, np.inner(sm, self.ndirection)/radius
    def construct_Links(self):
        self.FM = np.zeros((self.b,self.b), dtype = float)
        self.Q2CL = np.zeros((self.b, self.n), dtype = float)
        self.q_links = []
        for spin1 in xrange(self.nf - 1):
            for spin2 in xrange(spin1+1, self.nf):
                pos1 = tuple(self.FrTable[spin1])
                pos2 = tuple(self.FrTable[spin2])
                radius, cos_theta = self.get_smallest_vector(pos1, pos2)
                if (pos1 in self.q_spins) and (pos2 in self.q_spins):
                    val = self.coupling*(1-3*cos_theta**2)/radius**3*0.25
                    self.q_links.append((self.dTable[pos1], self.dTable[pos2], val))
                if xor((pos1 in self.q_spins), (pos2 in self.q_spins)):
                    if pos2 in self.q_spins:
                        pos1, pos2 = pos2, pos1
                    icl = self.cTable.index(self.FdTable[pos2])
                    iq = self.dTable[pos1]
                    val = self.coupling*(1-3*cos_theta**2)/radius**3*0.5
                    self.Q2CL[icl, iq] += val
                if (pos1 not in self.q_spins) and (pos2 not in self.q_spins):
                    val = self.coupling*(1-3*cos_theta**2)/radius**3
                    icl1 = self.cTable.index(spin1)
                    icl2 = self.cTable.index(spin2)
                    self.FM[icl2, icl1] += val
                    self.FM[icl1, icl2] += val
        self.CL2Q = self.Q2CL.transpose().copy()
        self.Q2CL *= self.Nroot
    def init_Hamiltonian(self):
        if self.hz[1] == 0.:
            dtype = np.float64
        else:
            dtype = np.complex128
        correction = (self.hz[0]!=0.)+(self.hz[1]!=0.)
        row = np.ndarray((self.N*(len(self.q_links) + self.n*correction + 1)), dtype = int)
        col = np.ndarray((self.N*(len(self.q_links) + self.n*correction + 1)), dtype = int)
        data = np.ndarray((self.N*(len(self.q_links) + self.n*correction + 1)), dtype = dtype)
        index = 0
        for vec in xrange(self.N):
            row[index], col[index], data[index] = vec, vec, 0
            for pos in xrange(self.n):
                val, vec2 = pauliZ(vec, pos)
                data[index] -= 0.5*self.hz[2]*val
            for link in self.q_links:
                sign = getSign(vec, link[0], link[1])
                data[index] -= sign*link[2]*self.coefs[2]
            if abs(data[index])<1e-9:
                data[index]=0.
            index += 1
            for link in self.q_links:
                vec2 = invert(vec, link[0], link[1])
                sign = getSign(vec, link[0], link[1])
                row[index], col[index], data[index] = vec2, vec, (sign*self.coefs[1]-self.coefs[0])*link[2]
                if abs(data[index])<1e-9:
                    data[index]=0.
                index+=1
            if self.hz[0]!=0.:
                for pos in xrange(self.n):
                    val, row[index] = pauli[0](vec, pos)
                    col[index], data[index] = vec, -0.5*val*self.hz[0]
                    index += 1
            if self.hz[1]!=0.:
                for pos in xrange(self.n):
                    val, row[index] = pauli[1](vec, pos)
                    col[index], data[index] = vec, -0.5*val*self.hz[1]
                    index += 1
        self.H = sparse.csr_matrix((data, (row, col)), shape=(self.N, self.N))
    def init_SpinOperators(self):
        self.taus = []
        for spin in xrange(self.n):
            self.taus.append([])
            for i in xrange(3):
                row = np.ndarray(self.N, dtype=int)
                col = np.ndarray(self.N, dtype=int)
                data = np.ndarray(self.N, dtype=complex)
                for vec in xrange(self.N):
                    data[vec], row[vec] = pauli[i](vec, spin)
                    col[vec] = vec
                self.taus[-1].append(sparse.coo_matrix((data, (row, col)), shape=(self.N, self.N)).tocsr())
    def init_Operators(self):
        def buildOperator(posList):
            size = self.N*len(posList)
            row = np.ndarray(size, dtype=int)
            col = np.ndarray(size, dtype=int)
            data = np.ndarray(size, dtype=complex)
            index = 0
            for vec in xrange(self.N):
                for pos in posList:
                    data[index], row[index] = pauli[0](vec,pos)
                    col[index] = vec
                    index += 1
            return sparse.coo_matrix((data, (row, col)), shape=(self.N, self.N)).tocsr()
        self.ops = {}
        posList = self.getMiddleSpins()
        self.ops['Local'] = buildOperator(posList)
        self.ops['Global'] = buildOperator(range(self.n))
    def setParameters(self, Tmax = None, delimiter = None):
        if delimiter is not None:
            self.delimiter = delimiter
        if Tmax is not None:
            self.Tmax = Tmax
            self.ts = np.arange(0, self.delimiter*self.Tmax+self.tstep, self.tstep)
    def initState(self, threshold = 2**(-10)):
        self.psi = self.randomState(threshold = threshold)
        self.clState = self.randomClassicalState()
        self.means = {}
        for op in self.ops.keys():
            self.means[op] = []
        self.cors = []
    def updateCorrelator(self):
        self.means['Local'].append(np.vdot(self.psi, self.ops['Local'].dot(self.psi)))
        temp = np.vdot(self.psi, self.ops['Global'].dot(self.psi))
        self.means['Global'].append(temp + np.sum(self.clState[0])*2*self.iNroot)
    def buildCors(self):
        start = 0
        for key in self.means.keys():
            self.means[key] = np.array(self.means[key])
        means1 = self.means[ 'Local']
        means2 = self.means['Global']
        delta = 0
        stop = len(means1)
        maxDelta = (stop - start)/self.delimiter
        while delta <= maxDelta:
            val = np.sum(means1[start:stop-1-delta]*means2[start+delta:stop-1])
            val += np.sum(means2[start:stop-1-delta]*means1[start+delta:stop-1])
            self.cors.append(val*0.5/float(stop-start-delta))
            delta += 1
        self.cors = np.array(self.cors)
    def calculateMeanFieldCoefs(self, psi):
        norm = np.vdot(psi, psi)
        for sigma in xrange(3):
            for spin in xrange(self.n):
                self.q_means[sigma, spin] = np.vdot(psi, self.taus[spin][sigma].dot(psi))/norm
    def rhs(self, psi, clState):
        self.calculateMeanFieldCoefs(psi)
        self.newPsi = self.H.dot(psi)
        for sigma in xrange(3):
            self.q_field[sigma] = self.CL2Q.dot(clState[sigma])*self.coefs[sigma]
            for spin in xrange(self.n):
                self.newPsi -= self.q_field[sigma, spin]*self.taus[spin][sigma].dot(psi)
            self.Hfield[sigma] = self.FM.dot(clState[sigma])*self.coefs[sigma]
            self.Hfield[sigma] += self.Q2CL.dot(self.q_means[sigma])*self.coefs[sigma]
            self.Hfield[sigma] += self.hz[sigma]
        self.newPsi *= -1j*self.tstep
        self.newClState[0] = clState[1]*self.Hfield[2] - clState[2]*self.Hfield[1]
        self.newClState[1] = clState[2]*self.Hfield[0] - clState[0]*self.Hfield[2]
        self.newClState[2] = clState[0]*self.Hfield[1] - clState[1]*self.Hfield[0]
        self.newClState *= self.tstep
        return self.newPsi, self.newClState
    def generateFileName(self, postfix = 'txt', **kwargs):
        converter = lambda x: x if x is not None else ""
        nTrials = kwargs.get('nTrials')
        Tmax = self.Tmax
        delimiter = self.delimiter
        sfile = "{prefix}D{D:d}d{d:d}n{n:d}fd{full_dims}H({Hx:.3f},{Hy:.3f},{Hz:.3f})Dir({Dx:.3f},{Dy:.3f},{Dz:.3f})Tmax{Tmax:.1f}Del{delimiter:d}nTr{nTrials:d}.{postfix}".format(prefix = self.classPrefix, D= self.D, d= self.d, n = self.n, full_dims = self.full_dims, \
                Hx = self.hz[0], Hy = self.hz[1], Hz = self.hz[2], Dx = self.direction[0], Dy = self.direction[1], Dz = self.direction[2], Tmax = Tmax, delimiter = converter(delimiter), nTrials = converter(nTrials), postfix = postfix)
        return sfile
    def plotData(self):
        plt.ion()
        plt.plot(self.ts[:len(self.cors)], self.cors/self.cors[0])


#########################################################


class FIDLatticeStub(HybridLatticeIR):
    classPrefix = "FIDLS"
    def __init__(self, D, lattice_dims, q_spins, middle_spins, coupling = 1., d = 1, cell_vectors = [], basis_vectors = None, hz = (0.,0.,0.), Tmax = 10., tstep = 0.0078125, delimiter = 10, direction = (1,0,0)):
        self.D = D
        self.d = d
        self.coupling = coupling
        assert isinstance(lattice_dims, tuple)
        assert D == len(lattice_dims)
        self.lattice_dims = lattice_dims
        self.full_dims = self.lattice_dims + (d,)
        self.cell_vectors = cell_vectors
        self.cell_vectors.append(tuple(np.zeros(D)))
        self.cell_vectors = np.array(self.cell_vectors)
        if basis_vectors is None:
            self.basis_vectors = np.eye(D)
        else:
            self.basis_vectors = np.array(basis_vectors)
        self.middle_spins = middle_spins
        self.q_spins = q_spins
        if d == 1:
            for i in xrange(len(self.middle_spins)):
                self.middle_spins[i] += (0,)
            for i in xrange(len(self.q_spins)):
                self.q_spins[i] += (0,)
        self.n = len(q_spins)
        self.N = 2**self.n
        self.nf = np.prod(self.full_dims)
        self.b = self.nf - self.n
        self.hz = hz
        self.direction = np.array(direction)
        self.ndirection = self.direction/nlinalg.norm(self.direction)
        self.Tmax = Tmax
        self.tstep = tstep
        self.delimiter = delimiter
        self.id6 = 1/6.
        self.id6 = 1/6.
        self.Nroot = np.sqrt(self.N+1)
        self.iNroot = 1/self.Nroot
        self.ts = np.arange(0, delimiter*Tmax+tstep, tstep)
        self.coefs = [-0.5,-0.5,1.]
        ###
        self.init_indexTables()
        self.generate_Increments()
        self.construct_Links()
        self.init_Hamiltonian()
        self.init_SpinOperators()
        self.initIntegrator()
        self.init_Operators()
        self.logfile = None
    def initIntegrator(self):
        self.psi = np.ndarray(self.N, dtype = complex)
        self.newPsi = np.ndarray(self.N, dtype = complex)
        self.clState = np.ndarray((3, self.b), dtype = float)
        self.Hfield = np.ndarray((3, self.b), dtype = float)
        self.newClState = np.ndarray((3, self.b), dtype = float)
        self.kq1 = np.ndarray(self.N, dtype = complex)
        self.kq2 = np.ndarray(self.N, dtype = complex)
        self.kq3 = np.ndarray(self.N, dtype = complex)
        self.kq4 = np.ndarray(self.N, dtype = complex)
        self.kcl1 = np.ndarray((3, self.b), dtype = float)
        self.kcl2 = np.ndarray((3, self.b), dtype = float)
        self.kcl3 = np.ndarray((3, self.b), dtype = float)
        self.kcl4 = np.ndarray((3, self.b), dtype = float)
        self.q_means = np.ndarray((3, self.n), dtype = float)
        self.q_field = np.ndarray((3, self.n), dtype = float)
    def generate_Increments(self):
        basis_increments = np.array([e for e in cartesian_product(*tuple([[-1,0,1]]*self.D))])
        bas_vec = self.basis_vectors.copy()
        for i in xrange(self.D):
            bas_vec[i] *= self.lattice_dims[i]
        self.deltas = basis_increments.dot(bas_vec)
    def get_vector(self, pos):
        vec = np.zeros((self.D,), dtype = float)
        for i in xrange(self.D):
            vec += pos[i]*self.basis_vectors[i]
        vec += self.cell_vectors[pos[-1]]
        return vec
    def get_smallest_vector(self, pos1, pos2):
        vec = self.get_vector(pos2) - self.get_vector(pos1)
        smallest = vec.copy()
        radius = nlinalg.norm(smallest)
        for delta in self.deltas:
            temp_vec = vec + delta
            temp_radius = nlinalg.norm(temp_vec)
            if temp_radius < radius:
                smallest = temp_vec.copy()
                radius = temp_radius
        #print pos1, pos2, smallest
        test_vec = np.ndarray((self.D + 1,), dtype = float)
        signs = np.sign(vec)
        for i in xrange(self.D):
            if abs(vec[i]) <= self.lattice_dims[i]/2.:
                signs[i] = 0
        test_vec = vec - signs*self.lattice_dims
        if not np.array_equal(test_vec, smallest):
            print pos1, pos2, smallest, test_vec
        if len(smallest)<len(self.ndirection):
            sm = smallest.copy()
            sm = np.append(sm, np.zeros(len(self.ndirection)-len(smallest)))
        else:
            sm = smallest
        return radius, np.inner(sm, self.ndirection)/radius
    def construct_Links(self):
        self.FM = np.zeros((self.b,self.b), dtype = float)
        self.Q2CL = np.zeros((self.b, self.n), dtype = float)
        self.q_links = []
        for spin1 in xrange(self.nf - 1):
            for spin2 in xrange(spin1+1, self.nf):
                pos1 = tuple(self.FrTable[spin1])
                pos2 = tuple(self.FrTable[spin2])
                radius, cos_theta = self.get_smallest_vector(pos1, pos2)
                if (pos1 in self.q_spins) and (pos2 in self.q_spins):
                    val = self.coupling*(1-3*cos_theta**2)/radius**3*0.25
                    self.q_links.append((self.dTable[pos1], self.dTable[pos2], val))
                if xor((pos1 in self.q_spins), (pos2 in self.q_spins)):
                    if pos2 in self.q_spins:
                        pos1, pos2 = pos2, pos1
                    icl = self.cTable.index(self.FdTable[pos2])
                    iq = self.dTable[pos1]
                    val = self.coupling*(1-3*cos_theta**2)/radius**3*0.5
                    self.Q2CL[icl, iq] += val
                if (pos1 not in self.q_spins) and (pos2 not in self.q_spins):
                    val = self.coupling*(1-3*cos_theta**2)/radius**3
                    icl1 = self.cTable.index(spin1)
                    icl2 = self.cTable.index(spin2)
                    self.FM[icl2, icl1] += val
                    self.FM[icl1, icl2] += val
        self.CL2Q = self.Q2CL.transpose().copy()
        self.Q2CL *= self.Nroot
    def init_Hamiltonian(self):
        if self.hz[1] == 0.:
            dtype = np.float64
        else:
            dtype = np.complex128
        correction = (self.hz[0]!=0.)+(self.hz[1]!=0.)
        row = np.ndarray((self.N*(len(self.q_links) + self.n*correction + 1)), dtype = int)
        col = np.ndarray((self.N*(len(self.q_links) + self.n*correction + 1)), dtype = int)
        data = np.ndarray((self.N*(len(self.q_links) + self.n*correction + 1)), dtype = dtype)
        index = 0
        for vec in xrange(self.N):
            row[index], col[index], data[index] = vec, vec, 0
            for pos in xrange(self.n):
                val, vec2 = pauliZ(vec, pos)
                data[index] -= 0.5*self.hz[2]*val
            for link in self.q_links:
                sign = getSign(vec, link[0], link[1])
                data[index] -= sign*link[2]*self.coefs[2]
            if abs(data[index])<1e-9:
                data[index]=0.
            index += 1
            for link in self.q_links:
                vec2 = invert(vec, link[0], link[1])
                sign = getSign(vec, link[0], link[1])
                row[index], col[index], data[index] = vec2, vec, (sign*self.coefs[1]-self.coefs[0])*link[2]
                if abs(data[index])<1e-9:
                    data[index]=0.
                index+=1
            if self.hz[0]!=0.:
                for pos in xrange(self.n):
                    val, row[index] = pauli[0](vec, pos)
                    col[index], data[index] = vec, -0.5*val*self.hz[0]
                    index += 1
            if self.hz[1]!=0.:
                for pos in xrange(self.n):
                    val, row[index] = pauli[1](vec, pos)
                    col[index], data[index] = vec, -0.5*val*self.hz[1]
                    index += 1
        self.H = sparse.csr_matrix((data, (row, col)), shape=(self.N, self.N))
    def init_SpinOperators(self):
        self.taus = []
        for spin in xrange(self.n):
            self.taus.append([])
            for i in xrange(3):
                row = np.ndarray(self.N, dtype=int)
                col = np.ndarray(self.N, dtype=int)
                data = np.ndarray(self.N, dtype=complex)
                for vec in xrange(self.N):
                    data[vec], row[vec] = pauli[i](vec, spin)
                    col[vec] = vec
                self.taus[-1].append(sparse.coo_matrix((data, (row, col)), shape=(self.N, self.N)).tocsr())
    def init_Operators(self):
        def buildOperator(posList):
            size = self.N*len(posList)
            row = np.ndarray(size*3, dtype=int)
            col = np.ndarray(size*3, dtype=int)
            data = np.ndarray(size*3, dtype=complex)
            index = 0
            for vec in xrange(self.N):
                for pos in posList:
                    for sigma in xrange(3):
                        data[index], row[index] = pauli[sigma](vec,pos)
                        data[index] *= self.ndirection[sigma]
                        col[index] = vec
                        index += 1
            return sparse.coo_matrix((data, (row, col)), shape=(self.N, self.N)).tocsr()
        self.ops = {}
        posList = self.getMiddleSpins()
        self.ops['Local'] = buildOperator(posList)
        self.ops['Global'] = buildOperator(range(self.n))
    def setParameters(self, Tmax = None, delimiter = None):
        if delimiter is not None:
            self.delimiter = delimiter
        if Tmax is not None:
            self.Tmax = Tmax
            self.ts = np.arange(0, self.delimiter*self.Tmax+self.tstep, self.tstep)
    def initState(self, threshold = 2**(-10)):
        self.psi = self.randomState(threshold = threshold)
        self.clState = self.randomClassicalState()
        self.means = {}
        for op in self.ops.keys():
            self.means[op] = []
        self.cors = []
    def updateCorrelator(self):
        self.means['Local'].append(np.vdot(self.psi, self.ops['Local'].dot(self.psi)))
        temp = np.vdot(self.psi, self.ops['Global'].dot(self.psi))
        #classical = 0
        #for sigma in xrange(3):
        #    classical += np.sum(self.clState[sigma])*self.ndirection[sigma]
        self.means['Global'].append(temp)# + classical*2*self.iNroot)
    def buildCors(self):
        start = 0
        for key in self.means.keys():
            self.means[key] = np.array(self.means[key])
        means1 = self.means[ 'Local']
        means2 = self.means['Global']
        delta = 0
        stop = len(means1)
        maxDelta = (stop - start)/self.delimiter
        while delta <= maxDelta:
            val = np.sum(means1[start:stop-1-delta]*means2[start+delta:stop-1])
            val += np.sum(means2[start:stop-1-delta]*means1[start+delta:stop-1])
            self.cors.append(val*0.5/float(stop-start-delta))
            delta += 1
        self.cors = np.array(self.cors)
    def calculateMeanFieldCoefs(self, psi):
        norm = np.vdot(psi, psi)
        for sigma in xrange(3):
            for spin in xrange(self.n):
                self.q_means[sigma, spin] = np.vdot(psi, self.taus[spin][sigma].dot(psi))/norm
    def rhs(self, psi, clState):
        #self.calculateMeanFieldCoefs(psi)
        self.newPsi = self.H.dot(psi)
        #for sigma in xrange(3):
        #    self.q_field[sigma] = self.CL2Q.dot(clState[sigma])*self.coefs[sigma]
        #    for spin in xrange(self.n):
        #        self.newPsi -= self.q_field[sigma, spin]*self.taus[spin][sigma].dot(psi)
        #    self.Hfield[sigma] = self.FM.dot(clState[sigma])*self.coefs[sigma]
        #    self.Hfield[sigma] += self.Q2CL.dot(self.q_means[sigma])*self.coefs[sigma]
        #    self.Hfield[sigma] += self.hz[sigma]
        self.newPsi *= -1j*self.tstep
        #self.newClState[0] = clState[1]*self.Hfield[2] - clState[2]*self.Hfield[1]
        #self.newClState[1] = clState[2]*self.Hfield[0] - clState[0]*self.Hfield[2]
        #self.newClState[2] = clState[0]*self.Hfield[1] - clState[1]*self.Hfield[0]
        #self.newClState *= self.tstep
        return self.newPsi, np.zeros_like(clState)#self.newClState
    def generateFileName(self, postfix = 'txt', **kwargs):
        converter = lambda x: x if x is not None else ""
        nTrials = kwargs.get('nTrials')
        Tmax = self.Tmax
        delimiter = self.delimiter
        sfile = "{prefix}D{D:d}d{d:d}n{n:d}fd{full_dims}H({Hx:.3f},{Hy:.3f},{Hz:.3f})Dir({Dx:.3f},{Dy:.3f},{Dz:.3f})Tmax{Tmax:.1f}Del{delimiter:d}nTr{nTrials:d}.{postfix}".format(prefix = self.classPrefix, D= self.D, d= self.d, n = self.n, full_dims = self.full_dims, \
                Hx = self.hz[0], Hy = self.hz[1], Hz = self.hz[2], Dx = self.direction[0], Dy = self.direction[1], Dz = self.direction[2], Tmax = Tmax, delimiter = converter(delimiter), nTrials = converter(nTrials), postfix = postfix)
        return sfile
    def plotData(self):
        plt.ion()
        plt.plot(self.ts[:len(self.cors)], self.cors/self.cors[0])


#########################################################


class FIDLatticeClassical(FIDLattice):
    classPrefix = "FIDLC"
    def __init__(self, D, lattice_dims, coupling = 1., d = 1, cell_vectors = [], basis_vectors = None, hz = (0.,0.,0.), Tmax = 10., tstep = 0.0078125, delimiter = 10, direction = (1,0,0)):
        self.D = D
        self.d = d
        self.coupling = coupling
        assert isinstance(lattice_dims, tuple)
        assert D == len(lattice_dims)
        self.lattice_dims = lattice_dims
        self.full_dims = self.lattice_dims + (d,)
        self.cell_vectors = cell_vectors
        self.cell_vectors.append(tuple(np.zeros(D)))
        self.cell_vectors = np.array(self.cell_vectors)
        if basis_vectors is None:
            self.basis_vectors = np.eye(D)
        else:
            self.basis_vectors = np.array(basis_vectors)
        #self.middle_spins = middle_spins
        #self.q_spins = q_spins
        #if d == 1:
        #    for i in xrange(len(self.middle_spins)):
        #        self.middle_spins[i] += (0,)
        #    for i in xrange(len(self.q_spins)):
        #        self.q_spins[i] += (0,)
        self.b = self.nf = np.prod(self.full_dims)
        self.hz = hz
        self.direction = np.array(direction)
        self.ndirection = self.direction/nlinalg.norm(self.direction)
        self.Tmax = Tmax
        self.tstep = tstep
        self.delimiter = delimiter
        self.id6 = 1/6.
        self.ts = np.arange(0, delimiter*Tmax+tstep, tstep)
        self.coefs = [-0.5,-0.5,1.]
        ###
        self.init_indexTables()
        self.generate_Increments()
        self.construct_Links()
        #self.init_Hamiltonian()
        #self.init_SpinOperators()
        self.initIntegrator()
        #self.init_Operators()
        self.logfile = None
    def initIntegrator(self):
        self.clState = np.ndarray((3, self.b), dtype = float)
        self.Hfield = np.ndarray((3, self.b), dtype = float)
        self.newClState = np.ndarray((3, self.b), dtype = float)
        self.k1 = np.ndarray((3, self.b), dtype = float)
        self.k2 = np.ndarray((3, self.b), dtype = float)
        self.k3 = np.ndarray((3, self.b), dtype = float)
        self.k4 = np.ndarray((3, self.b), dtype = float)
    def init_indexTables(self):
        self.FdTable = np.arange(self.nf).reshape(self.full_dims)
        self.FrTable = np.ndarray((self.nf,len(self.full_dims)),dtype=int)
        for index, val in np.ndenumerate(self.FdTable):
           self.FrTable[val] = np.array(index)
    def construct_Links(self):
        self.FM = np.zeros((self.b,self.b), dtype = float)
        for spin1 in xrange(self.b - 1):
            for spin2 in xrange(spin1+1, self.b):
                pos1 = tuple(self.FrTable[spin1])
                pos2 = tuple(self.FrTable[spin2])
                radius, cos_theta = self.get_smallest_vector(pos1, pos2)
                val = self.coupling*(1-3*cos_theta**2)/radius**3
                self.FM[spin2, spin1] += val
                self.FM[spin1, spin2] += val
    def initState(self, threshold = 2**(-10)):
        self.clState = self.randomClassicalState()
        self.means = []
        self.cors = []
    def updateState(self):
        self.rungeKuttaStep(self.clState)
    def updateCorrelator(self):
        self.means.append(np.sum(self.clState[0]))
    def buildCors(self, start = 0, delimiter = 10):
        means = np.array(self.means)
        delta = 0
        stop = len(means)
        maxDelta = (stop - start)/self.delimiter
        while delta <= maxDelta:
            val = np.sum(means[start:stop-1-delta]*means[start+delta:stop-1])
            self.cors.append(val/float(stop-start-delta))
            delta += 1
        self.cors = np.array(self.cors)
    def rhs(self, clState):
        for sigma in xrange(3):
            self.Hfield[sigma] = self.FM.dot(clState[sigma])*self.coefs[sigma]
            self.Hfield[sigma] += self.hz[sigma]
        self.newClState[0] = clState[1]*self.Hfield[2] - clState[2]*self.Hfield[1]
        self.newClState[1] = clState[2]*self.Hfield[0] - clState[0]*self.Hfield[2]
        self.newClState[2] = clState[0]*self.Hfield[1] - clState[1]*self.Hfield[0]
        self.newClState *= self.tstep
        return self.newClState
    def rungeKuttaStep(self, clState):
        super(HybridLattice, self).rungeKuttaStep(clState)
    def generateFileName(self, postfix = 'txt', **kwargs):
        converter = lambda x: x if x is not None else ""
        nTrials = kwargs.get('nTrials')
        Tmax = self.Tmax
        delimiter = self.delimiter
        sfile = "{prefix}D{D:d}d{d:d}fd{full_dims}H({Hx:.3f},{Hy:.3f},{Hz:.3f})Dir({Dx:.3f},{Dy:.3f},{Dz:.3f})Tmax{Tmax:.1f}Del{delimiter:d}nTr{nTrials:d}.{postfix}".format(prefix = self.classPrefix, D= self.D, d= self.d, full_dims = self.full_dims, \
                Hx = self.hz[0], Hy = self.hz[1], Hz = self.hz[2], Dx = self.direction[0], Dy = self.direction[1], Dz = self.direction[2], Tmax = Tmax, delimiter = converter(delimiter), nTrials = converter(nTrials), postfix = postfix)
        return sfile


#########################################################

class FIDLatticeClStub(FIDLatticeClassical):
    classPrefix = "FIDLCS"
    def __init__(self, D, lattice_dims, coupling = 1., d = 1, cell_vectors = [], basis_vectors = None, hz = (0.,0.,0.), Tmax = 10., tstep = 0.0078125, delimiter = 10, direction = (1,0,0), coefs = [-0.41,-0.41,0.82]):
        self.D = D
        self.d = d
        self.coupling = coupling
        assert isinstance(lattice_dims, tuple)
        assert D == len(lattice_dims)
        self.lattice_dims = lattice_dims
        self.full_dims = self.lattice_dims + (d,)
        self.cell_vectors = cell_vectors
        self.cell_vectors.append(tuple(np.zeros(D)))
        self.cell_vectors = np.array(self.cell_vectors)
        if basis_vectors is None:
            self.basis_vectors = np.eye(D)
        else:
            self.basis_vectors = np.array(basis_vectors)
        #self.middle_spins = middle_spins
        #self.q_spins = q_spins
        #if d == 1:
        #    for i in xrange(len(self.middle_spins)):
        #        self.middle_spins[i] += (0,)
        #    for i in xrange(len(self.q_spins)):
        #        self.q_spins[i] += (0,)
        self.b = self.nf = np.prod(self.full_dims)
        self.hz = hz
        self.direction = np.array(direction)
        self.ndirection = self.direction/nlinalg.norm(self.direction)
        self.Tmax = Tmax
        self.tstep = tstep
        self.delimiter = delimiter
        self.id6 = 1/6.
        self.ts = np.arange(0, delimiter*Tmax+tstep, tstep)
        self.coefs = coefs
        ###
        self.init_indexTables()
        self.generate_Increments()
        self.construct_Links()
        #self.init_Hamiltonian()
        #self.init_SpinOperators()
        self.initIntegrator()
        #self.init_Operators()
        self.logfile = None
    def construct_Links(self):
        self.FM = np.zeros((self.b,self.b), dtype = float)
        for spin1 in xrange(self.b - 1):
            for spin2 in xrange(spin1+1, self.b):
                pos1 = tuple(self.FrTable[spin1])
                pos2 = tuple(self.FrTable[spin2])
                radius, cos_theta = self.get_smallest_vector(pos1, pos2)
                print(spin1,spin2,radius, radius==1.0)
                if radius==1.0:
                    print "Yes\n"
                    val = 1.0
                else:
                    val = 0.
                self.FM[spin2, spin1] += val


#########################################################


class ClassicalLattice(HybridLattice):
    classPrefix = "cLl"
    def __init__(self, dims, hz = (0.,0.,0.), Js = (-0.41,-0.41,0.82), Tmax = 10., tstep = 0.0078125, delimiter = 10, Axes = [0,1,2]):
        self.dims = dims
        self.n = np.prod(dims)
        self.N = 2**self.n
        self.D = len(dims)
        self.hz = hz
        self.Js = np.array(Js)
        self.Tmax = Tmax
        self.tstep = tstep
        self.delimiter = delimiter
        self.ts = np.arange(0, delimiter*Tmax+tstep, tstep)
        self.id6= 1/6.
        self.setAxes(Axes)
        self.init_indexTables()
        self.generate_Increments()
        self.construct_Links()
        self.init_Integrator()
        self.logfile = None
    def setAxes(self, Axes):
        self.axes = Axes
    def randomClassicalState(self):
        state = np.ndarray((3, self.n), dtype = float)
        zs = 2*np.random.rand(self.n) - 1
        rs = np.sqrt(1-np.square(zs))
        angles = np.random.rand(self.n)*2*np.pi
        state[0] = rs*np.cos(angles)
        state[1] = rs*np.sin(angles)
        state[2] = zs
        state *= np.sqrt(3)/2.
        #state.transpose()
        return state
    def setParameters(self, Tmax = None, delimiter = None, Axes = None):
        if delimiter is not None:
            self.delimiter = delimiter
        if Tmax is not None:
            self.Tmax = Tmax
            self.ts = np.arange(0, self.delimiter*self.Tmax+self.tstep, self.tstep)
        if Axes is not None:
            self.setAxes(Axes)
    def init_indexTables(self):
        super(ClusteredLattice, self).init_indexTables()
    def init_Integrator(self):
        self.clState = np.ndarray((3, self.n), dtype = float)
        self.newClState = np.ndarray((3, self.n), dtype = float)
        self.Hfield = np.ndarray((3, self.n), dtype = float)
        self.k1 = np.ndarray((3, self.n), dtype = float)
        self.k2 = np.ndarray((3, self.n), dtype = float)
        self.k3 = np.ndarray((3, self.n), dtype = float)
        self.k4 = np.ndarray((3, self.n), dtype = float)
    def construct_Links(self):
        self.FM = np.zeros((self.n, self.n), dtype = int)
        for pos in xrange(self.n):
            for delta in self.deltas:
                for sign in [-1,1]:
                    newIndex = (self.rTable[pos] + sign*delta) % self.dims
                    newPos = self.dTable[tuple(newIndex)]
                    self.FM[pos, newPos] += 1
        self.FM = sparse.csr_matrix(self.FM)
    def initState(self):
        self.clState = self.randomClassicalState()
        self.means = {}
        for axis in self.axes:
            self.means[axis] = []
        self.cors = []
        for i in xrange(len(self.axes)):
            self.cors.append([])
    def updateState(self):
        self.rungeKuttaStep(self.clState)
    def updateCorrelator(self):
        for axis in self.axes:
            self.means[axis].append(np.sum(self.clState[axis]))
    def buildCors(self, start = 0, delimiter = 10):
        for i in xrange(len(self.axes)):
            means = np.array(self.means[self.axes[i]])
            delta = 0
            stop = len(means)
            maxDelta = (stop - start)/self.delimiter
            while delta <= maxDelta:
                val = np.sum(means[start:stop-1-delta]*means[start+delta:stop-1])
                self.cors[i].append(val/float(stop-start-delta))
                delta += 1
        self.cors = np.array(self.cors)
    def rhs(self, clState):
        for sigma in xrange(3):
            self.Hfield[sigma] = self.FM.dot(clState[sigma])*self.Js[sigma]
            self.Hfield[sigma] += self.hz[sigma]
        self.newClState[0] = clState[1]*self.Hfield[2] - clState[2]*self.Hfield[1]
        self.newClState[1] = clState[2]*self.Hfield[0] - clState[0]*self.Hfield[2]
        self.newClState[2] = clState[0]*self.Hfield[1] - clState[1]*self.Hfield[0]
        self.newClState *= self.tstep
        return self.newClState
    def rungeKuttaStep(self, clState):
        super(HybridLattice, self).rungeKuttaStep(clState)
    def propagate(self, clState = None):
        self.elapsedTime = -time.time()
        self.initState()
        if clState is not None:
            self.clState = clState.copy()
        t = 0.
        Tstop = self.ts[-1]
        self.updateCorrelator()
        while t  < Tstop:
            t += self.tstep
            self.updateState()
            self.updateCorrelator()
        self.buildCors()
        self.elapsedTime += time.time()
        self.output(self.elapsedTime)
    def generateFileName(self, postfix = 'txt', **kwargs):
        converter = lambda x: x if x is not None else ""
        nTrials = kwargs.get('nTrials')
        Tmax = self.Tmax
        delimiter = self.delimiter
        axs = ""
        for axis in self.axes:
           axs  += str(axis) +','
        axs = axs[:-1]
        sfile = "{prefix}n{n:d}d{dims}H({Hx:.3f},{Hy:.3f},{Hz:.3f})J({Jx:.3f},{Jy:.3f},{Jz:.3f})Tmax{Tmax:.1f}Del{delimiter:d}nTr{nTrials:d}A{axes}.{postfix}".format(prefix = self.classPrefix, n = self.n, dims = self.dims, \
                Hx = self.hz[0], Hy = self.hz[1], Hz = self.hz[2], Jx = self.Js[0], Jy = self.Js[1], Jz = self.Js[2], Tmax = Tmax, delimiter = converter(delimiter), nTrials = converter(nTrials), axes = axs, postfix = postfix)
        return sfile



