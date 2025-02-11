from scipy.special import ellipk, ellipe, dawsn
import matplotlib.pyplot as plt
import numpy as np
import Real_parametrs as rp

N0 = 2**13
e = 4.8*1e-10 # единицы сгс
m = 9.1*1e-28 #г ~ 10^-28 г
m_eff = 0.635*m
lambd0 = 744*1e-7 #см ~ 800 нм
H_plank = 1.054*1e-27 #эрг*с
c = 3*1e10 #см/с ~ 10^8 m/s
w_0 = 2*np.pi*c/lambd0 #c^-1
sigma = 8*1e-19
delta =  9*1.6*1e-12 # band gap energy
I_at = 3.1*1e+23 # Atomic_itensity (эрг/см^2) ~ 31000 ТВт
E_at = np.sqrt(8*np.pi*I_at/c)
w_at = 41.3*1e15 # Atomic_frequency (1/s)
r_h = delta/(13.6*1.6*1e-12) # Ionization_coef.()
N_a = 2.2*1e22
#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
I0 =  1.11764333434023*1e16# erg/cm^2*c
tau = 50*1e-15
t = np.linspace(-300*1e-15, 300*1e-15, N0)
# t = rp.t*1e-15
#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
I = 30*1e19 # pulse Itensity 10TW/cm3 = 10^12*10^7 erg/s*cm2
A = np.sqrt(8*np.pi*I/c)*np.exp(-0.5*(t/tau)**2)

E = A*np.cos(w_0*t)


# E = rp.E_env
# A = np.sqrt(8*np.pi*I0/c) * E

h = t[1] - t[0]
y = w_0*np.sqrt(m_eff*delta)/(e*A)

Hamma = y*y/(1+y*y)
ksi = 1/(1+y*y)
eKH = ellipk(Hamma)
eKH[np.isinf(eKH)] = eKH[~np.isinf(eKH)].max()
alpha = np.pi*(eKH - ellipe(Hamma))/ellipe(ksi)

x = (2*delta/(np.pi*H_plank*w_0))*ellipe(ksi)/np.sqrt(Hamma)

nu = np.floor(x+1)-x

row = 0
betta = np.pi**2/(4*ellipk(ksi)*ellipe(ksi))

# for n in range(3000):
#     row += exp(-n*alpha)*dawsn(sqrt(betta*(n+2*nu)))

n = np.arange(3000)[:, None]
row = (np.exp(-n*alpha)*dawsn(np.sqrt(betta*(n+2*nu)))).sum(axis=0)


Q = np.sqrt(np.pi/(2*ellipk(ksi)))*row

W_E_ = ((w_0*m_eff)/(H_plank*np.sqrt(Hamma)))**1.5
W_E_ *= Q*np.exp(-alpha*np.floor(x+1))
W_E_ *= (2*w_0/(9*np.pi))

W_PI = W_E_/N_a
#-----------------------------------------------------------------------------------------
def Ne(A, no_aval = False):
    e_osc = ((e*A)**2)/(4*m*w_0**2)
    e_cr = 1.5*(delta + e_osc)
    K = np.array(e_cr/(H_plank*w_0), dtype = int)
    #K = np.floor(array(e_cr/(H_plank*w_0))).astype('int')
    etta = sigma/(((2**(1/K) - 1))*e_cr)
    W_pt = etta*A**2*c/(8*np.pi)
    # W_pt = 0.*A
    h = t[1] - t[0]
    N = np.zeros((max(K), N0))
    
    if no_aval:
        W_pt *= 0
        
    for j in range(1, N0):
        for i in range(K[j]):
            if i == 0:
                N[0, j] = h*(W_PI[j-1] - W_pt[j-1]*N[0, j-1] + 
                              2*W_pt[j-1]*N[K[j]-1, j-1]) + N[0, j-1]
            else:
                N[i, j] = h*(W_pt[j-1]*N[i-1, j-1] - W_pt[j-1]*N[i, j-1]) + N[i, j-1]
        if K[j]  < K[j-1]:
            # N[K[j], j] += N[K[j], j-1]
            # N[K[j], j-1] = 0
            N[K[j]-1, j] += N[K[j-1]-1, j-1]
            N[K[j-1]-1, j] = 0
    #-----------------------------------------------------------------------------------------    
    #-----------------------------------------------------------------------------------------
    
    N = N.sum(axis=0)
    return N

def r_der(arr, dx):
    der = [0, 0]
    for i in range(len(arr)-2):
        der.append((-3*arr[i] + 4*arr[i+1] - arr[i+2])/(2*dx))
    return np.array(der)

Der_Ne_FLD = (r_der(Ne(A), h))
Der_Ne_ENVLP = (r_der(Ne(A, no_aval = True), h))

der_new = W_PI*(1 - Ne(A))

J_1 = Der_Ne_FLD/A
for i in range(N0):
    if abs(A[i]) < 10000:
        J_1[i] = 0

J_2 = Der_Ne_ENVLP/A
for i in range(N0):
    if abs(A[i]) < 10000:
        J_2[i] = 0
        
J_3 = Der_Ne_ENVLP/E
for i in range(N0):
    if abs(E[i]) < 10000:
        J_3[i] = 0


J_4 = der_new/A

for i in range(N0):
    if t[i] < 0:
        J_4[i] = J_1[i]

# # J_4 += J_4[::-1]

#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
t2 = t*1e15

plt.figure(figsize=(15, 10))

# plt.plot(t2, Der_Ne_FLD, "b.", t2, Der_Ne_ENVLP, "r", t2, der_new, "g--", linewidth = 4)
plt.plot(t2, J_1, "b.", t2, J_2, "r", t2, J_4, "g--", linewidth = 2)
# plt.plot(t2, Ne(A), t2, Ne(A, no_aval = True))

plt.rcParams['font.size'] = '23'
# plt.plot(t, A*A*(c/8*pi)/1e7)
plt.xlabel("Время (с)", size = 27)
plt.legend(["С лавиной" , "Без лавины", "Получен из лавины"])
# plt.ylabel("Плотность электронов(-)", size = 27)
# plt.title(f'{I/1e19} TW/cm^2')
plt.grid(True)
# plt.xlim([0.3*1e-13, 0.9*1e-13])
plt.show()
