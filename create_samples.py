import os,sys,pdb
import numpy as np
import random

class SAMPLEGEN(object):
    def __init__(self, prPi, prT, prO, maxT = 100): #pi, T, O
        self.save_prPi_prT_prO(prPi, prT, prO)
        self.numS = prPi.size
        self.numO = prO.shape[1]
        assert self.numS == prO.shape[0] and self.numS == prT.shape[0] and self.numS == prT.shape[1]
        self.cdfPi = self.getCFD(prPi) #hiddne status
        #transion Pr
        self.cdfT = np.zeros( (self.numS, self.numS + 1)   )
        for row in range(self.numS):
            self.cdfT[row,:] = self.getCFD(prT[row,:])
        #emit Pr
        self.cdfO = np.zeros((self.numS, self.numO+1)) #observation #status * #observation
        for row in range(self.numS):
            self.cdfO[row,:] = self.getCFD(prO[row,:])
        self.minT = 3
        self.maxT = np.maximum(self.minT, maxT)
        return
    def save_prPi_prT_prO(self, prS, prT, prO):
        prS = np.reshape(prS, (1,-1))
        prT = np.reshape(prT, (prS.shape[1],-1) )
        prO = np.reshape(prO, (prS.shape[1],-1))
        lines = []
        lines.append('-----prPi-------')
        line = ','.join( ['%.3f'%x for x in prS.tolist()[0]] )
        lines.append(line)
        lines.append('------prT------')
        for row in range(prT.shape[0]):
            line = ','.join( ['%.3f'%x for x in prT[row,:].tolist() ]   )
            lines.append(line)
        lines.append('-------prO-----')
        for row in range(prO.shape[0]):
            line = ','.join( ['%.3f'%x for x in prO[row,:].tolist()  ]  )
            lines.append(line)
        with open('prS_prT_prO.txt','wb') as f:
            f.write('\r\n'.join(lines))
        return

    def getCFD(self, pr):
        if isinstance(pr,list):
            pr = np.asarray(pr)
        pr = np.reshape(pr,(1,-1)) / pr.sum()
        cdf = np.zeros((1, 1 + pr.shape[1]))
        cdf[0,0] = 0
        prev = 0
        for k in range(pr.shape[1]):
            cdf[0,k + 1] = prev + pr[0,k]
            prev += pr[0,k]
        return cdf
    def random_selection(self,cdf):
        pr = random.random()
        return (pr>cdf).sum() - 1
    def gen_one_seq(self):
        T = random.randint(self.minT,self.maxT)
        seqS = []
        seqO = []
        lastS = self.random_selection(self.cdfPi)
        O = self.random_selection(self.cdfO[lastS,:])
        seqS.append(lastS)
        seqO.append(O)
        for t in range(1,T,1):
            S = self.random_selection(self.cdfT[lastS,:])
            O = self.random_selection(self.cdfO[S,:])
            lastS = S
            seqS.append(S)
            seqO.append(O)
        return seqS,seqO
    def run(self,outfile, N):
        outputs = []
        for n in range(N):
            seqS,seqO = self.gen_one_seq()
            seqS = ','.join( ['%d'%x for x in seqS] )
            seqO = ','.join( ['%d'%x for x in seqO] )
            outputs.append( '|'.join(['%d'%n,seqS,seqO]) ) # id | status | observation
        with open(outfile, 'wb') as f:
            f.writelines('\r\n'.join(outputs))

if __name__=="__main__":
    numS = 2
    numO = 2
    prS = np.random.random( (1, numS))
    prT = np.random.random( (numS,numS))
    prO = np.random.random((numS, numO))
    prS /= prS.sum()

    sums = np.tile( np.reshape( prT.sum(axis=1), (numS,-1)), (1,numS) )
    prT = prT / sums

    sums = np.tile( np.reshape( prO.sum(axis=1), (numS,-1)), (1,numO))
    prO = prO / sums

    gen = SAMPLEGEN(prS, prT,prO, 10)
    gen.run('samples.txt',1000)





