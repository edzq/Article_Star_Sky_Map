import numpy
import sklearn.metrics as metrics
import copy
# 使用自己定义的tsne算法，算法原理与python sklearn中的TSNE相同，细节上稍微不同
def cal_matrix_P(X,neighbors):
    """
    X:输入的高维矩阵
    neighbors:相邻点的个数
    
    return:高维中分布密度矩阵
    """
    entropy=numpy.log(neighbors)
    n1,n2=X.shape
    D=numpy.square(metrics.pairwise_distances(X))
    D_sort=numpy.argsort(D,axis=1)
    P=numpy.zeros((n1,n1))
    for i in range(n1):
        Di=D[i,D_sort[i,1:]]
        P[i,D_sort[i,1:]]=cal_p(Di,entropy=entropy)
    P=(P+numpy.transpose(P))/(2*n1)
    P=numpy.maximum(P,1e-100)
    return P


def cal_p(D,entropy,K=50):
    """
    D:其他点相对于某个点的距离，从小到大排序
    
    return：其他点相对与这个点的密度分布
    """
    beta=1.0
    H=cal_entropy(D,beta)
    error=H-entropy
    k=0
    betamin=-numpy.inf
    betamax=numpy.inf
    while numpy.abs(error)>1e-4 and k<=K:
        if error > 0:
            betamin=copy.deepcopy(beta)
            if betamax==numpy.inf:
                beta=beta*2
            else:
                beta=(beta+betamax)/2
        else:
            betamax=copy.deepcopy(beta)
            if betamin==-numpy.inf:
                beta=beta/2
            else:
                beta=(beta+betamin)/2
        H=cal_entropy(D,beta)
        error=H-entropy
        k+=1
    P=numpy.exp(-D*beta)
    P=P/numpy.sum(P)
    return P


def cal_entropy(D,beta):
    """
    return：计算其他点相对于某个点分布的熵/密度的一种衡量方式
    """
    # P=numpy.exp(-(numpy.sqrt(D))*beta)
    P=numpy.exp(-D*beta)
    sumP=sum(P)
    sumP=numpy.maximum(sumP,1e-200)
    H=numpy.log(sumP) + beta * numpy.sum(D * P) / sumP
    return H


def cal_matrix_Q(Y):
    """
    Y：低维向量
    
    return：低维向量分布密度矩阵
    """
    n1,n2=Y.shape
    D=numpy.square(metrics.pairwise_distances(Y))
    #Q=1/(1+numpy.exp(D))
    #Q=1/(1+numpy.square(D))
    #Q=1/(1+2*D)
    #Q=1/(1+0.5*D)
    Q=(1/(1+D))/(numpy.sum(1/(1+D))-n1)
    Q=Q/(numpy.sum(Q)-numpy.sum(Q[range(n1),range(n1)]))
    Q[range(n1),range(n1)]=0
    Q=numpy.maximum(Q,1e-100)
    return Q

## 改变损失函数

def cal_gradients(P,Q,Y,Y_original,I):
    """
    P:高维向量分布密度矩阵
    Q:低维坐标分布矩阵
    Y:低维坐标
    Y_original:初始低维向量/可以随机，也可以使用上一次降维的结果
    I:偏离参数，调整与原先的坐标偏离的容忍程度,I 越小，容忍程度越低。
    
    return:下降的梯度
    """
    n1,n2=Y.shape
    DC=numpy.zeros((n1,n2))
    for i in range(n1):
        E=(1+numpy.sum((Y[i,:]-Y)**2,axis=1))**(-1)
        F=Y[i,:]-Y
        G=(P[i,:]-Q[i,:])
        E=E.reshape((-1,1))
        G=G.reshape((-1,1))
        G=numpy.tile(G,(1,n2))
        E=numpy.tile(E,(1,n2))
        if i < len(Y_original):
            DC[i,:]=numpy.sum(4*G*E*F,axis=0)+2*I*(numpy.sqrt(numpy.square(Y[i,:]-Y_original[i,:])))
        else:
            DC[i,:]=numpy.sum(4*G*E*F,axis=0)            
    return DC

def cal_loss(P,Q,Y,Y_original,I):
    """
    return:损失函数
    """
    C=numpy.sum(P * numpy.log(P / Q))+I*numpy.sum(numpy.square(Y[:len(Y_original)]-Y_original))
    return C


def tsne(X,Y_original,n=2,neighbors=30,I=0.001,max_iter=200):
    """
    X:高维向量
    Y_original:低维向量的初始值
    n:=Y_original.shape[1],低维的维度
    neighbors:邻居个数，这个会在计算分布密度熵的时候，作为一个标准
    I:偏离参数，与原先的坐标偏离的容忍程度，I 越小，容忍程度越高。
    max_iter:最大的迭代次数
    
    return:低维坐标
    """
    #tsne_dat=shelve.open('tsne.dat')
    data=[]
    n1,n2=X.shape
    P=cal_matrix_P(X,neighbors)
    Y=numpy.random.randn(n1,n)
    #Y=numpy.random.randn(n1,n)
    for i in range(len(Y_original)):
        Y[i] = Y_original[i]
    Q = cal_matrix_Q(Y)
    DY = cal_gradients(P, Q, Y, Y_original, I)
    A=200.0
    B=0.1
    for i in range(max_iter):
        data.append(Y)
        if i==0:
            Y=Y-A*DY
            Y1=Y
            error1=cal_loss(P, Q, Y, Y_original, I)
        elif i==1:
            Y=Y-A*DY
            Y2=Y
            error2=cal_loss(P, Q, Y, Y_original, I)
        else:
            YY=Y-A*DY+B*(Y2-Y1)
            QQ = cal_matrix_Q(YY)
            error=cal_loss(P, QQ, Y, Y_original, I)
            if error>error2:
                A=A*0.7
                continue
            elif (error-error2)>(error2-error1):
                A=A*1.2
            Y=YY
            error1=error2
            error2=error
            Q = QQ
            DY = cal_gradients(P, Q, Y, Y_original, I)
            Y1=Y2
            Y2=Y
        if cal_loss(P, Q, Y, Y_original, I)<1e-4:
            print(cal_loss(P, Q, Y, Y_original, I))
            return Y
        if numpy.fmod(i+1,10)==0:
            print ('%s iterations the error is %s, A is %s'%(str(i+1),str(round(cal_loss(P, Q, Y, Y_original, I),2)),str(round(A,3))))
    #tsne_dat['data']=data
    #tsne_dat.close()
    return Y