import numpy as np
from sklearn import preprocessing
from sklearn import datasets
import matplotlib.pyplot as plt
# @Zhiwei

def preprocess(X,Y):
    # 处理数据集X和Y
    scaler_X = preprocessing.StandardScaler().fit(X)
    scaler_Y = preprocessing.StandardScaler().fit(Y)
    X_mean=scaler_X.mean_
    Y_mean=scaler_Y.mean_
    X_mean=X_mean.reshape((X_mean.shape[0],1))

    X_dot=np.array(X.T)-np.array(X_mean)
    X_L2=np.sqrt(np.sum(X_dot*X_dot,1)).reshape((X_mean.shape[0],1))
    X_dot=X_dot/X_L2
    X=np.mat(X_dot)

    Y_dot=np.array(Y.T)-np.array(Y_mean.T)
    Y=np.mat(Y_dot)

    return X.T,Y.T

def process_Y(Y):
    # sklearn里的糖尿病数据X标准化，Y没有标准化
    scaler_Y = preprocessing.StandardScaler().fit(Y)
    Y_mean = scaler_Y.mean_
    Y_dot = np.array(Y.T) - np.array(Y_mean)
    Y = np.mat(Y_dot)
    return Y.T

def cal_u(X_A):
    # 计算角平分向量
    g_A=X_A.T*X_A
    one_A=np.mat(np.ones((np.array(X_A).shape[1],1)))
    A_A=float(1/np.sqrt(one_A.T*g_A.I*one_A))
    w_A=A_A*(g_A.I*one_A)
    u_A=X_A*w_A
    print("A_A:")
    print(A_A)
    print("w_A:")
    print(w_A)
    return A_A,w_A,u_A

def cal_c(y):
    # 计算残差与X.T的相关度
    return X.T * y

# 输出sign（）处理的X_A
def cal_X_A(c,A):
    #从X中提取X_A，A为有效集
    X_A=X[:,A]
    sj=np.sign(c)
    sj=sj[A,:]
    X_A=np.array(X_A.T)*np.array(sj)
    X_A=np.mat(X_A).T
    return X_A,sj

def cal_C(c):
    # C为最大相关度
    C=abs(float(c[np.argmax(np.abs(c), axis=0)]))
    return C

def cal_a(X,u_A):
    return X.T*u_A

# A_C为基于X的A补集的索引
def cal_gama(C,c,a,A_A,A_C):
    data=np.zeros((len(A_C),2))
    count=0
    for index in A_C:
        temp_left=(C-float(c[index]))/(A_A-float(a[index]))
        if(temp_left<0):
            temp_left=100000000
        temp_right=(C+float(c[index]))/(A_A+float(a[index]))
        if (temp_right < 0):
            temp_right = 1000000000
        data[count][0]=temp_left
        data[count][1]=temp_right
        count=count+1
    print("gama_matrix:")
    print(data)
    index = data.argmin(axis=0)
    if (data[index[0]][0] > data[index[1]][1]):
        gama = data[index[1]][1]
        j = index[1]
    else:
        gama = data[index[0]][0]
        j = index[0]
    j=A_C[j]
    print("gama:")
    print(gama)
    return gama,j

def cal_A_C(X,A):
    feature_num=X.shape[1]
    A_C=[]
    for i in range(feature_num):
        if(i in A):
            continue
        else:
            A_C.append(i)
    return A_C
def update_Beta(gama,Beta,A,w_A,sj):
    Beta[A,:]=Beta[A,:]+gama*np.mat(np.array(w_A)*np.array(sj))
    return Beta

def draw(beta_list,feature_num):
    # 画出beta变化趋势图
    x=list(range(0,feature_num+1))
    fig, ax = plt.subplots()
    ax.set_ylim([-800.,800.])

    for i in range(feature_num):
        ax.plot(x, beta_list[i, :])
    plt.show()

def lars(X,Y):
    feature_num=X.shape[1]
    Beta=np.zeros((feature_num,1))
    Miu_A = np.mat(np.zeros((n,1)))
    # 初始化μ为0
    A=[]
    beta_list=np.zeros((feature_num,1))
    y=Y-Miu_A
    c=X.T*y
    j=np.argmax(np.abs(c))
    C=float(abs(c[j]))
    # C为一个float型的数字
    A.append(j)

    for i in range(feature_num):
        X_A,sj=cal_X_A(c,A)
        # print("第{}次X_A：{}".format(i,X_A))
        A_A,w_A,u_A=cal_u(X_A)
        c=cal_c(y)
        print("c:")
        print(c)
        C=cal_C(c)
        print("C:")
        print(C)

        a=cal_a(X,u_A)
        A_C=cal_A_C(X,A)
        if(i<feature_num-1):
            gama, j = cal_gama(C, c, a, A_A, A_C)
        else:
            gama=C/A_A
        Beta=update_Beta(gama,Beta,A,w_A,sj)
        print("Beta:")
        print(Beta)
        beta_list=np.append(beta_list,Beta,1)
        Miu_A=Miu_A+gama*u_A
        y=Y-Miu_A
        print("A:")
        print(A)
        if(j not in A):
            A.append(j)
            print("-------------------------------------------")
    print("打印总的beta变化矩阵：")
    # 一行代表一个beta的变化
    # 一列代表所处当前迭代次数的所有beta的值
    print(beta_list)
    draw(beta_list,feature_num)
    # return Beta




if __name__=='__main__':

    diabetes = datasets.load_diabetes()
    X = diabetes.data
    Y = diabetes.target

    Y = Y.reshape((Y.shape[0], 1))
    n = X.shape[0]
    # 加载的数据X已经标准化符合要求
    # X, Y = preprocess(X, Y)
    Y = process_Y(Y)
    X = np.mat(X)
    Y = np.mat(Y)
    lars(X, Y)

















