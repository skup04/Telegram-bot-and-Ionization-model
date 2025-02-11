from scipy.special import ellipk, ellipe, dawsn
import matplotlib.pyplot as plt
from numpy import pi, sqrt, exp, linspace, floor, arange, log10
from Real_parametrs import E_env, t

e = 4.8*1e-10 # единицы сгс
m = 9.1*1e-28 #г ~ 10^-28 г
m_eff = 0.635*m
lambd0 = 744*1e-7 #см ~ 800 нм
H_plank = 1.054*1e-27 #эрг*с
c = 3*1e10 #см/с ~ 10^8 m/s
w_0 = 2*pi*c/lambd0 #c^-1
sigma_e = 8*1e-19
#-----------------------------------------------------------------------------------------
N0 = 2**12

E_env += 1

# I = linspace(16, 22, N0)
I = linspace(16, 22, N0)
I = 10**I
delta =  9*1.6*1e-12 # band gap energy
A = sqrt(8*pi*I/c)
N_a = 2.2*1e22

y = w_0*sqrt(m_eff*delta)/(e*sqrt(8*pi*I/c))


Hamma = y*y/(1+y*y)
ksi = 1 - Hamma

alpha = pi*(ellipk(Hamma) - ellipe(Hamma))/ellipe(ksi)

x = (2*delta/(pi*H_plank*w_0))*ellipe(ksi)/sqrt(Hamma)


nu = floor(x+1)-x

row = 0
betta = pi*pi/(4*ellipk(ksi)*ellipe(ksi))

# for n in range(3000):
#     row += exp(-n*alpha)*dawsn(sqrt(betta*(n+2*nu)))

n = arange(3000)[:, None]
row = (exp(-n*alpha)*dawsn(sqrt(betta*(n+2*nu)))).sum(axis=0)


Q = sqrt(pi/(2*ellipk(ksi)))*row

# W_E_ = zeros(N0)
# for i in range(N0):
#     W_E_[i] = (2*w_0/(9*pi))*(((w_0*m_eff)/(H_plank*Hamma[i]))**1.5)*Q[i]*exp(-alpha[i]*int(x[i]+1))

W_E_ = ((w_0*m_eff)/(H_plank*sqrt(Hamma)))**1.5
W_E_ *= Q*exp(-alpha*floor(x+1))
W_E_ *= (2*w_0/(9*pi))


plt.figure(figsize=(15, 10))
# plt.loglog(I*1e-7, W_E_, 'ro-')
plt.loglog(I*1e-7, W_E_/N_a, 'r-')
# plt.semilogy(I*1e-7, W_E_)
# plt.xlim([1e12, 1e13])
# plt.ylim([1e25, 1e38])
plt.grid(True)
plt.xlabel("Интенсивность (Вт/см^2)", fontsize = 20)
plt.ylabel("Скорость ионизации (с^-1*см^-3)", fontsize = 20)
W_E_ = log10(W_E_)
