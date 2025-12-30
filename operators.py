import qutip as qp
import numpy as np
from scipy.integrate import quad

def construct_Hamiltonian(g, w, N, mu, m):
    sx_list = [] # i-th element is sigma x performing on (i+1)-th qubit
    sy_list = [] # i-th element is sigma y performing on (i+1)-th qubit
    sz_list = [] # i-th element is sigma z performing on (i+1)-th qubit
    
    for n in range(N):
        ope_list = [qp.qeye(2)]*N # a list of identity operator
        ope_list[n] = qp.sigmax() # replace the i-th element by sigma x
        sx_list.append(qp.tensor(ope_list))
        
        ope_list = [qp.qeye(2)]*N # a list of identity operator
        ope_list[n] = qp.sigmay() # replace the i-th element by sigma y
        sy_list.append(qp.tensor(ope_list))

        ope_list = [qp.qeye(2)]*N # a list of identity operator
        ope_list[n] = qp.sigmaz() # replace the i-th element by sigma z
        sz_list.append(qp.tensor(ope_list))
    
    #Changed
    # First term
    Hzz = 0
    for n in range(1, N-1):
        for l in range(n+1):
            for k in range(l):
                Hzz += sz_list[k]*sz_list[l]
    Hzz = (g/(8*w))*Hzz

    # Second term
    Hxxyy = 0
    for n in range(N-1):
        Hxxyy += sx_list[n]*sx_list[n+1] + sy_list[n]*sy_list[n+1]
    Hxxyy = (w/(2*g))*Hxxyy

    #Changed
    # Third term
    Hz1 = 0
    for n in range(N):
        Hz1 += ((m/g)*((-1)**(n+1)) + (mu/g))*sz_list[n]
    Hz1 = Hz1/2

    #Changed
    # Forth term
    Hz2 = 0
    for n in range(N-1):
        for l in range(n+1):
            Hz2 += ((n+1)%2)*sz_list[l]
    Hz2 = -(g/(8*w))*Hz2

    H = Hzz + Hxxyy + Hz1 + Hz2
    return H

def get_exact_Gibbs_state(H, T, g):
    rho = (-1*H*g/T).expm()
    tr = rho.tr()
    rho = rho/tr
    return rho

def est_chiral(N,rho,a):
    

    sz_list = [] # i-th element is sigma z performing on (i+1)-th qubit
    for n in range(N):
        ope_list = [qp.qeye(2)]*N # a list of identity operator
        ope_list[n] = qp.sigmaz() # replace the i-th element by sigma z
        sz_list.append(qp.tensor(ope_list))
    
    ope = 0
    #Changed
    for n in range(N):
        ope += (1/(2*N*a))*((-1)**(n+1))*sz_list[n]
    
    value = abs((rho*ope).tr())

    return value

def f_function(x, a):
    fx = 1/(1 - np.exp(a*np.cosh(x)))
    return fx

def I_function(a):
    Ia = quad(f_function, 0, np.inf, args = (a, ))
    return Ia[0]

def theo_chiral(T, mu, g):
    m_gamma = g/np.sqrt(np.pi)
    beta = 1/T
    gamma = 0.57721
    cal_chiral = (-m_gamma/(2*np.pi))*np.exp(gamma)*np.exp(2*I_function(beta*m_gamma))
    abs_chiral = abs(cal_chiral)
    return abs_chiral