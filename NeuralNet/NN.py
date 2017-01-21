import numpy as np


def prepare_dataset(dataset):
    data=[]
    with open(dataset) as R:
        for line in R:
            line=line.split()
            #print float (line[0].strip("',"))
            data.append([float(i.strip("',")) for i in line])
    return np.array(data, dtype=np.float)

def readTrainSet(TrainD, TrainY):
    dataT=[]
    targetT=[]
    with open(TrainD) as TD, open(TrainY) as TY:
        for line in TD:
            line=line.split()
            dataT.append([float(i.strip("',"))  for i in line])
        for line in TY:
            targetT.append(int(line))
    return (np.array(dataT),np.array(targetT))


def Initialization(F,HL,HN,O):
    """
    M=[]
    Initialize a Matrix of dimension FxHN
    Add to M
    For each HL-1
        Initialize a Random Matrix of Size HNxHN
        Add to M
    Initialize a Random Matrix of Size HNxO
    Add to M
    """
    M=[]
    Layer1=np.random.rand(F,HN)
    M.append(Layer1)
    for i in range(0,HL-1):
        Layer=np.random.rand(HN,HN)
        M.append(Layer)
    layerO=np.random.rand(HN,O)
    M.append(layerO)
    return M


def sigmoid(X):
    return 1.0/(1-np.exp(-X))

def der_sigmoid(X):
    return (sigmoid(X)*(1-sigmoid(X)))

def forwardpass( M,TrainT):
    Fs=[]
    Fd=[]
    N,F=TrainT.shape

    L=len(M)
    """
    for i in range(L):
        F1=np.dot(TrainT,M[i])
        Fs.append(F1)
    """
    T=TrainT
    #print T.shape
    for i in range(L):
       # print T.shape, M[i].shape
        F1=sigmoid(np.dot(T,M[i]))


        #print F1
        Fs.append(F1)
        #print F1.shape
        F2=np.transpose(der_sigmoid(F1))
        #print F2.shape
        Fd.append(F2)
        T=F1
    #print T.shape


    return (Fs,Fd)


def squared_error(Predicted, Target):
    error=np.sum(np.square(Target-Predicted))*0.5
    return error
def delta(predicted,target):
    return (target-predicted)

def backwardpass(delta,M,Fd):
    """

    :param delta:
    :param M:
    :param Fs:
    :return:
    """
    Ds=[]
    l=len(M)
    d=np.transpose(delta)
    Ds.append(d)

    for i in range(1,l):
        #print d.shape,M[-i].shape
        T=np.dot(M[-i],d)

        #print T.shape
        D=np.multiply(Fd[-i-1],T)
        #print D.shape
        d=D
        Ds.append(d)
    for i in Ds:
        print i.shape
    return Ds


def weight_update(X,Fs,M,Ds,eta=0.9):
    '''

    :param X:
    :param Fs:
    :param M:
    :param Ds:
    :param eta:
    :return:
    '''
    #Please complete the code here. Please also write
    # the details of your computation







def Training(params,TrainT, TrainY):
    N=params["N"]
    F=params['Input']
    HL=params["HLayers"]
    HN=params["HNodes"]
    O=params["Output"]
    M=Initialization(F,HL,HN,O)
    #print params
    #for each iteration in range(1,1000):
    Fs,Fd=forwardpass(M,TrainT)
    Error=squared_error(np.floor(Fs[-1]),TrainY)
    d=delta(np.floor(Fs[-1]),TrainY)
    print 'i', d.shape
    Ds=backwardpass(d,M,Fd)
    #M=weight_update(TrainT,Fs,M,Ds,0.9)
    print M


def NN(parameters, TrainT, TrainY):

    Training(parameters,TrainT,TrainY)
    #error=(target-predict)
    pass

def demo_input():
    return np.array([[1 ,2] ,[1,3],[2,3]])
def demo_target():
    return np.array([[1],[0],[1]])
if __name__ == '__main__':

    #
    #dataT,targetT=readTrainSet("TrainSet.txt","TrainY.txt")
    #print targetT.shape, dataT.shape
    #print list(targetT)
    dataT=demo_input()
    print dataT
    dataY=demo_target()
    N,F=dataT.shape
    O=1
    hl=3
    hn=5
    parameters={"N":N,"Input":F,"Output":O,"HLayers":hl,"HNodes":hn}

    NN(parameters,dataT,dataY)
    #print delta(np.array([[1,2],[1,2]]), np.array([[1,2],[2,1]]))
    #print squared_error(np.array([[1,2],[1,2]]), np.array([[1,2],[2,1]]))

    #M=Initialization(2,3,5,2)
    #print len(M)
    #print M[3]
    #print 1-M[3]






