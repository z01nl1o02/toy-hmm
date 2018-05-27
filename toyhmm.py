import os,sys,pdb,cPickle
import numpy as np

class TOYHMM(object):
    def __init__(self, numS, numO, trainFlag=True, verbose = False):
        self.trainFlag = trainFlag
        self.verbose = verbose
        if numS < 0 or numO < 0: #call load() after init()
            self.numS, self.numO = 0, 0
            self.prPi, self.prT, self.prO = None,None,None
        else:
            self.prPi = np.zeros((1,numS))
            self.prT = np.zeros((numS, numS))
            self.prO = np.zeros((numS, numO))
            self.numS = numS
            self.numO = numO
        return

    def init_guess(self):
        self.prPi = np.random.random(self.prPi.shape)
        self.prPi /= self.prPi.sum()

        self.prT = np.random.random(self.prT.shape)
        sums = np.tile( np.reshape(self.prT.sum(axis=1),(self.numS,-1)), (1, self.numS)  )
        self.prT = self.prT / sums


        self.prO = np.random.random(self.prO.shape)
        sums = np.tile( np.reshape( self.prO.sum(axis=1), (self.numS,-1)) , (1,self.prO.shape[1]) )
        self.prO = self.prO / sums;
        return
    # meanings of forwad and backword
    # A.prPi (1 x numS)
    # B. prT (numS x numS)
    # C. prPi x prT.transpose() -> prS (1 x numS)
    # D. prS.transpose() x prO[:,o] -> (numS x 1)
    # ABCD is four steps in HMM. splitting it to two parts
    # forward part: prPi x prT.transpose() -> prS
    # backward part: prT.transpose() x prO[:.o].reshape((1, numS))
    # the reason to split like this is forward x backward equal to ABCD !!!
    def calc_forward_pr(self,seqO): #P(o_0,o_1,...,o_t|S_t)
        eps = 0.0001
        T = len(seqO)
        outputPr = np.zeros( (self.numS,T) )
        prevPr = self.prPi
        for t in range(T):
            o = seqO[t]
            outputPr[:,t] = prevPr.reshape((self.numS,)) * self.prO[:,o]
            if self.trainFlag:
                outputPr[:,t] /= outputPr[:,t].sum()
            prevPr = self.prT.transpose().dot( outputPr[:,t].reshape(self.numS,1)  )
        return outputPr
    def calc_backward_pr(self, seqO, initPr = 1.0): #P(o_{t+1}, o_{t_2}, ...,o_{T}|S_{t+1})
        eps = 0.0001
        T = len(seqO)
        outputPr = np.zeros( (self.numS,self.numS,T)  ) #(currentS, nextS, t) it is "prT" in time
        nextPr = self.prT
        for t in range(T-1,0,-1):
            o = seqO[t]
            outputPr[:,:,t] = nextPr.transpose() * self.prO[:,o].reshape((1,-1))
            if self.trainFlag:
                ss = outputPr[:,:,t].sum(axis = 1)
                ss = np.tile( ss.reshape(-1,1), (1,numS)  )
                outputPr[:,:,t] /= ss
            nextPr = nextPr * self.prT
        return outputPr
    def train_EStep(self,seqO):
        prF = self.calc_forward_pr(seqO)
        prB = self.calc_backward_pr(seqO)
        return prF, prB

    def train_MStep(self, seqO, prF, prB):
        eps = 0.00001
        #pr(S)
        prPiRaw = np.reshape( prF.sum(axis = 1), self.prPi.shape )
        prPi = prPiRaw / prPiRaw.sum()

        #pr(O|S)
        prO = np.zeros(self.prO.shape)
        for t,o in enumerate(seqO):
            prO[:,o] += np.reshape( prF[:,t], prO[:,o].shape )
        prO[ prO < eps  ] = eps
        prO = prO / np.tile( np.reshape( prPiRaw, (self.numS,1)), (1, self.numO)  )

        #pr(S2 | S1)
        prT = np.zeros( self.prT.shape )
        for t in range(prF.shape[1]):
            m = prB[:,:,t] * np.tile( prF[:,t].reshape(-1,1), (1, self.numS) )
            prT += m
        m = np.tile( prT.sum(axis=1).reshape(-1,1), (1,self.numS) )
        prT = prT / m
        return (prPi, prT, prO)
    def train_one(self,seqO):
        th = 0.001
        #self.init_guess()
        while True:
            prF, prB = self.train_EStep(seqO)
            prPi, prT, prO = self.train_MStep(seqO, prF, prB)

            dfPi = np.absolute(prPi - self.prPi).sum()
            dfT = np.absolute(prT - self.prT).sum()
            dfO = np.absolute(prO - self.prO).sum()
            if dfPi < th  and dfT < th and dfO < th:
                break
            #print dfPi, dfT,dfO
            self.prPi, self.prT, self.prO = prPi, prT, prO
        return self.prPi, self.prT, self.prO

    def train(self, multSeqO):
        prPi, prT, prO = [],[],[]
        self.init_guess()
        if self.verbose:
            print 'train start'
        for num,seqO in enumerate(multSeqO):
            a,b,c = self.train_one(seqO)
            prPi.append(a)
            prT.append(b)
            prO.append(c)
            if self.verbose:
              print '%d\t'%(num+1),
        if self.verbose:
            print 'train done'
        if 0:
            prPi = prPi[-1]
            prT = prT[-1]
            prO = prO[-1]
        else:
            prPi = reduce(lambda x,y:x+y, prPi) / len(multSeqO)
            prT = reduce(lambda x,y:x+y, prT) / len(multSeqO)
            prO = reduce(lambda x,y:x+y, prO) / len(multSeqO)
        self.prPi, self.prT, self.prO = prPi, prT, prO
        return self.prPi, self.prT, self.prO

    def save(self,filepath):
        with open(filepath,'wb') as f:
            cPickle.dump( (self.numS, self.numO, self.prPi, self.prT, self.prO), f   )
        return

    def load(self, filepath):
        with open(filepath, 'rb') as f:
            self.numS, self.numO, self.prPi, self.prT, self.prO = cPickle.load(f)
        return

    def evaluate(self, seqO): #given observation, to get its Pr from the model
        prF = self.calc_forward_pr(seqO)
        return prF[:, len(seqO)-1].sum()

    def decode(self, seqO): #give observation, to get status to generate it with maximum Pr
        prF = self.calc_forward_pr(seqO)
        seqS = []
        for t in range(len(seqO)):
            seqS.append(  np.argmax(prF[:,t])  )
        return seqS

def next_sample(infile):
    with open(infile,'rb') as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            idx,seqS,seqO = line.split('|')
            seqS = [np.int64(x) for x in seqS.split(',')]
            seqO = [np.int64(x) for x in seqO.split(',')]
            idx = np.int64(idx)
            yield  idx,seqS, seqO
    return

def train(samplefile, modelfile, numS, numO):
    trainRatio = 0.1
    multSeqO = []
    for idx, seqS, seqO in next_sample(samplefile):
        multSeqO.append(seqO)
    num = np.int64( len(multSeqO) * trainRatio )
    #num = 1
    multSeqO = multSeqO[0:num]
    print 'train with %d seq'%len(multSeqO)
    hmm = TOYHMM(numS, numO,trainFlag = False, verbose = True)
    hmm.train(multSeqO)
    hmm.save(modelfile)
    return

def test(samplefile, modelfile):
    hmm = TOYHMM(-1,-1,trainFlag=False)
    hmm.load(modelfile)
    prlist = []
    misslist = []
    lines = []
    for idx, seqS, seqO in next_sample(samplefile):
        pr = hmm.evaluate(seqO)
        seqS1 = hmm.decode(seqO)
        miss = reduce(lambda x,y: x + y, [np.int64(x != y) for x,y in zip(seqS, seqS1)]) * 1.0 / len(seqS)
        lines.append( ','.join( ['%d'%x for x in seqS] ) )
        lines.append( ','.join( ['%d'%x for x in seqS1] ) )
        lines.append('\r\n')
        prlist.append(pr)
        misslist.append(miss)
    with open('test.log','wb') as f:
        f.writelines('\r\n'.join(lines))
    prlist = np.asarray(prlist)
    misslist = np.asarray(misslist)
    print 'pr: ',prlist.mean(), prlist.std()
    print 'miss: ',misslist.mean(), misslist.std()
    return


if __name__=="__main__":
    numS, numO = 2,2
    samplefile = 'samples.txt'
    modelfile = 'hmm.pkl'
    if sys.argv[1] == 'train':
        train(samplefile, modelfile, numS, numO)
        test(samplefile, modelfile)
    elif sys.argv[1] == 'test':
        test(samplefile, modelfile)
    else:
        print 'unk option'



















