import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy import constants

# данные Земли
# m0 = 287000 + 4725 #сухая масса
# M = 258000 # масса топлива
# Ft = 3268861.02
# Cf = 0.5
# ro = 1.293 # плотность воздуха
# S = constants.pi * ((10.3/2)**2)
# g = constants.g
# k = M/314.5 #скорость расхода топлива (через 314.5 сек отсоединилась последняя ступень)


# данные Кербина
m0 = 46904 
M = m0 + 41482 # масса с топливом
Ft = 3268861.02
Cf = 0.5
ro = 1.293 # плотность воздуха
S = constants.pi * ((6.6/2)**2)
g = 1.00034 * constants.g
k = (M-m0)/(187) # скорость расхода топлива (через 187 сек отсоединилась последняя ступень)

def F1(t):
    return (Ft/(M - k*t))

def F2(t):
    return ((Cf*ro*S)/(2*(M - k*t)))

def dv_dt(t, v):
    return (F1(t) - F2(t)*v**2 - g) 

v0 = 0
t = np.linspace(0, 34, 1080) 

solution = solve_ivp(dv_dt, t_span = (0, max(t)), y0 = [v0], t_eval = t)

x = solution.t
y = solution.y[0]

plt.figure(figsize=(7, 6))
plt.plot(x, y, '-r', label="v(t)")
plt.legend()
plt.grid(True)
plt.show()
