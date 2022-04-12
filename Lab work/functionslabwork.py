import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import scipy
from scipy.integrate import ode

def Curant(N, T, L, M):
    tau = T/N
    h = L/M
    sigma = tau/h**2
    print(f'Параболическое число Куранта',sigma)
    
def u_exect(M, N, L, T):
    u_th = np.zeros((M, N))
    x = [i for i in np.arange(0, L, L/M)]
    t = [i for i in np.arange(0, T, T/N)]
    for i in range(M):
        for j in range(N):
            u_th[i][j]=x[i]*(1+4*t[j])**(-3/2)*np.exp(-x[i]**2/(1+4*t[j]))
    return u_th

def num_sol(M, N, L, T, ksi):
    t = [i for i in np.arange(0, T, T/N)]
    x = [i for i in np.arange(0, L, L/M)]
    h = L/M
    tau = T/N
    u = np.zeros((M, N))
    #Заполнение 1 столбца массива численного решения из НУ
    for m in range(M):
        u[m][0] = x[m]*np.exp(-x[m]**2)
    #построение численного решения
    A = np.zeros((M, M))
    A[0][0] = 1
    P = -ksi/h**2
    Q = 1/tau+2*ksi/h**2
    R = (1-ksi)/h**2
    S = 1/tau-2*(1-ksi)/h**2
    for i in range (1, M-1):
        A[i][i-1] = P
        A[i][i] = Q
        A[i][i+1] = P
    A[M-1][M-1] = 1

    f = np.zeros((M, N)) 
    for n in range(N):
        f[M-1][n] = L*(1+4*t[n-1])**(-3/2) * np.exp(-L**2/(1+4*t[n-1]))
        for m in range(1, M-1):
            f[m][n] = u[m-1][n-1]*R + u[m][n-1]*S + u[m+1][n-1]*R

    u_cur = np.linalg.pinv(A) @ f

    for i in range(1, N):
        for j in range(M):
            u[j][i] = u_cur[j][i]
    return u


def plot_3d(u):
    fig = go.Figure(data=[
        go.Surface(z= u)])
    fig.update_layout(scene = dict(
                    xaxis_title='t',
                    yaxis_title='x',
                    zaxis_title=f'u'),
                    width=700,
                    margin=dict(r=20, b=10, l=10, t=10))
    fig.show()
    
def nev(N, T, u_th, u):
    t = [i for i in np.arange(0, T, T/N)]
    delta = [0 for j in range(N)] #вектор невязки
    for n in range(N):
        delta[n] = np.linalg.norm(u_th[:, n] - u[:, n])
    plt.plot(t, delta)
    plt.xlabel('t')
    plt.ylabel('delta')
    plt.show()
    
def nev1(N, T, u_th, u):
    t = [i for i in np.arange(0, T, T/N)]
    delta = [0 for j in range(N)] #вектор невязки
    delta[1] = 0
    for n in range(1, N):
        delta[n] = np.linalg.norm(u_th[:, n] - u[:, n])
    return delta


def numer_sol(T, L, M, N):
    h = L/M
    tau = T/N
    t = np.arange(0, T, tau)
    x = np.arange(0, L, h)
    
    ut = np.zeros(M)
    for m in range(M): 
        ut[m] = x[m]*np.exp(-(x[m])**2)

    r = scipy.integrate.ode(func_du).set_integrator("dopri5")  #ode23tb(@eq6, [0 T], ut);
    r.set_initial_value(ut, 0)

    y = np.zeros((M, len(t)))
    for i in range(1, len(t)):
        y[:, i] = r.integrate(t[i]) # get one more value, add it to the array
        if not r.successful():
            raise RuntimeError("Could not integrate")
    for m in range(M): 
        y[m][0] = x[m]*np.exp(-(x[m])**2)
    
    return y

def func_du(t, u):
    L = 1
    M = 7
    h = L/M
    du = [0 for i in range(M)]
    du[1]=(0- 2*u[2]+u[3])/h**2
    for m in range(2, M-2):
        du[m] = (u[m-1]-2*u[m]+u[m+1])/h**2
    du[M-2]=(u[M-3]-2*u[M-2]+L*(1+4*t)**(-3/2)*np.exp(-L**2/(1+4*t)))/h**2
    du[M-1]=((4*L**3)*np.exp(-L**2/(1+4*t)))/(1+4*t)**(7/2) - (6*L*np.exp(-L**2/(1+4*t)))/(1+4*t)**(5/2)
    return(du)
