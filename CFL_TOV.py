import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
pi=np.pi

Msolar=1.98847e30 #Solar Mass in SI (kg) Units
c=2.99792458e8 #Speed of light in SI (m/s) Units
G=6.67430e-11 #Gravitational Constant in SI Units

####### !!In the following we use Geometrized Units ( we express our quanities in M(ega)Meters^n ) for convenience.!! #######
####### !!The conversion to SI Units follows appropriately.                                                        !! #######




#Define the TOV equation 

def dmdr(r,P):
    eps=3*P+4*Beff+(3*w**2/(4*pi**2))+(w/pi)*np.sqrt(3*(P+Beff)+(9*w**2/(16*pi**2))) #The Equation of State
    dmdr=4*pi*eps*r**2
    return dmdr

def dPdr(r,m,P):
    eps=3*P+4*Beff+(3*w**2/(4*pi**2))+(w/pi)*np.sqrt(3*(P+Beff)+(9*w**2/(16*pi**2)))
    dPdr=-(eps+P)*(m+4*np.pi*r**3*P)/(r*(r-2*m))
    return dPdr





#The RK4 Method

def RK4_TOV(f,g,Pc,r1,rmax,h):
    m=[0]
    P=[Pc]
    r=[0]
    epsc=3*Pc+4*Beff+(3*w**2/(4*pi**2))+(w/pi)*np.sqrt(3*(Pc+Beff)+(9*w**2/(16*pi**2)))
    m1=(4/3)*np.pi*epsc*r1**3
    P1=Pc-2.0*pi*(epsc+Pc)*(Pc+epsc/3)*r1**2.0
    if P1>=0:
        m.append(m1)
        P.append(P1)
        r.append(r1)
        n=int((rmax-r1)/h)
        for i in range(1,n+1):
            k1=h*f(r1,P1)
            l1=h*g(r1,m1,P1)
            k2=h*f(r1+h/2,P1+l1/2)
            l2=h*g(r1+h/2,m1+k1/2,P1+l1/2)
            k3=h*f(r1+h/2,P1+l2/2)
            l3=h*g(r1+h/2,m1+k2/2,P1+l2/2)
            k4=h*f(r1+h,P1+l3)
            l4=h*g(r1+h,m1+k3,P1+l3)
            m1=m1 + 1/6 * (k1+2*k2+2*k3+k4)
            P1=P1 + 1/6 * (l1+2*l2+2*l3+l4)
            if P1<0:
                break
            m.append(m1)
            P.append(P1)
            r1=r1+h
            r.append(r1)
    eps=[3*i+4*Beff+(3*w**2/(4*pi**2))+(w/pi)*np.sqrt(3*(i+Beff)+(9*w**2/(16*pi**2))) for i in P]
    M=m[-1]
    R=r[-1]
    return [M,R]




rmax=0.1 #providing a high limit on the Radii
h=2e-7   #provide an appropriate step
r1=h

### The pressure in the core (Pc) has to be chosen appropriately by considering the sitffness/softness of the EoS which is fixed by Beff and w ###
### The softer the EoS (e.g., higher Beff) the higher the limit on Pc ###
### Example sets follow below ###


#Pc0=np.logspace(-7,1.9,100)
#Pc1=np.logspace(-7,2.2,200)
#Pc2=np.logspace(-5,2.5,200)
Pc3=np.logspace(-5,2.7,100) 
#Pc4=np.logspace(-3.5,2.85,120)
#Pc5=np.logspace(-7,2.3,120)
M=[]
R=[]

Beff=100 #insert desired value here#
w=0 #and here#
for k in range(0,len(Pc3)):
            sol=RK4_TOV(dmdr,dPdr,Pc3[k],r1,rmax,h)
            M.append(sol[0])
            R.append(sol[1])

M=np.array(M)
R=np.array(R)

#Convert to SI units#

M=M*1.3466e33/Msolar
R=R*10**3


#Plot the results#

plt.figure()
plt.plot(R,M)
plt.ylim(bottom=0.1,top=2.85)
plt.xlabel('$R \, [km]$',fontsize=16)
plt.ylabel('$M/M_\odot$',fontsize=16)
plt.minorticks_on()
plt.tick_params(direction='in',right='True',top='True',which='both')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
