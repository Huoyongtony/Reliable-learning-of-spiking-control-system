import numpy as np
import matplotlib.pyplot as plt
import time
from numba import jit,njit,typeof
from numba import int32, float32 ,float64,vectorize   # import the types
from numba.experimental import jitclass
from numba.typed import List as NumbaList
from scipy.integrate import solve_ivp
from typing import List
import pickle 
exp=np.exp

@vectorize([float64(float64, float64,float64)])
def sinf(V,Vth,Vslope):
    if V < Vth:
        sinf = 0
    else:
        sinf = np.tanh((V-Vth)/Vslope)
    return sinf
    
#sinf2(V::Float64) = boltz(V,25.,-5.)
@jit
def sinf2(V):
    return boltz(V,25.,-5.)
@jit 
def max_abs(a,b):
    if abs(a)>abs(b):
        return a
    else:
        return b
    
    
@jit 
def heaviside (t):
    return (1+np.sign(t))/2
@jit 
def pulse (t,ti,tf):
    return heaviside(t-ti)-heaviside(t-tf)

sin=np.sin
pi=np.pi
#boltz(V::Float64,A::Float64,B::Float64) = 1/(1 + exp((V+A)/B))
@jit
def boltz(V,A,B):
    return 1/(1 + exp((V+A)/B))
#tauX(V::Float64,A::Float64,B::Float64,D::Float64,E::Float64) = A - B/(1+exp((V+D)/E))
@jit
def tauX(V,A,B,D,E):
    return A - B/(1+exp((V+D)/E))
#mNainf(V::Float64) = boltz(V,25.5,-5.29)
    
@jitclass([("mis",float64[:,:]),("mis_t",typeof(np.ones((12,4))*0.1))])   
class dyns:
    
    
    def __init__(self,mis,mis_t,mismatch_time=False,mismatch_act=False):
        if mismatch_act:
            self.mis=mis
        else:
            self.mis=np.ones((12,2)) 
            
        if mismatch_time:
            self.mis_t=mis_t
        else:
            self.mis_t=np.ones((12,4)) 
           
        
            

    def mNa_inf(self,V):
        return boltz(V,25.5*self.mis[0][0],-5.29*self.mis[0][1])
    #taumNa(V::Float64) = tauX(V,1.32,1.26,120.,-25.)
 
    def tau_mNa(self,V):
        return tauX(V,1.32*self.mis_t[0][0],1.26*self.mis_t[0][1],120.*self.mis_t[0][2],-25.*self.mis_t[0][3])
    #hNainf(V::Float64) = boltz(V,48.9,5.18)

    def hNa_inf(self,V):
        return boltz(V,48.9*self.mis[1][0],5.18*self.mis[1][1])
    #tauhNa(V::Float64) = (0.67/(1+exp((V+62.9)/-10.0)))*(1.5 + 1/(1+exp((V+34.9)/3.6)))

    def tau_hNa (self,V):
        return (0.67/(1+exp((V+62.9*self.mis_t[1][0])/-10.0*self.mis_t[1][2])))*(1.5*self.mis_t[1][1] + 1/(1+exp((V+34.9*self.mis_t[1][2])/3.6*self.mis_t[1][3])))
    #mCaTinf(V::Float64) = boltz(V,27.1,-7.2)
    
    def mt_inf (self,V):
        return boltz(V,27.1*self.mis[2][0],-7.2*self.mis[2][1])
    #taumCaT(V::Float64,taumCa::Float64) = taumCa*tauX(V,21.7,21.3,68.1,-20.5)
    
    def tau_mt (self,V,taumCa):
        return taumCa*tauX(V,21.7*self.mis_t[2][0],21.3*self.mis_t[2][1],68.1*self.mis_t[2][2],-20.5*self.mis_t[2][3])
    #hCaTinf(V::Float64) = boltz(V,32.1,5.5)
    
    def ht_inf(self,V):
        return boltz(V,32.1*self.mis[3][0],5.5*self.mis[3][1])
    #tauhCaT(V::Float64) = tauX(V,105.,89.8,55.,-16.9)
    
    def tau_ht(self,V):
        return tauX(V,105.*self.mis_t[3][0],89.8*self.mis_t[3][1],55.*self.mis_t[3][2],-16.9*self.mis_t[3][3])
    #mCaSinf(V::Float64) = boltz(V,33.,-8.1)
    
    def mS_inf(self,V):
        return boltz(V,33.*self.mis[4][0],-8.1*self.mis[4][1])
    #taumCaS(V::Float64,taumCa::Float64) = taumCa*(1.4 + (7/((exp((V+27)/10))+(exp((V+70)/-13)))))
    
    def tau_mS(self,V,taumCa):
        return taumCa*(1.4*self.mis_t[4][0] + (7*self.mis_t[4][1]/((exp((V+27*self.mis_t[4][2])/10*self.mis_t[4][3]))+(exp((V+70*self.mis_t[4][0])/-13*self.mis_t[4][1])))))
    #hCaSinf(V::Float64) = boltz(V,60.,6.2)
    
    
    def hS_inf(self,V):
        return boltz(V,60.*self.mis[5][0],6.2*self.mis[5][1])
    
    def tau_hS(self,V):
        mis_t=self.mis_t
        return 60*mis_t[5][0] + (150*mis_t[5][1]/((exp((V+55*mis_t[5][2])/9*mis_t[5][3]))+(exp((V+65*mis_t[5][0])/-16*mis_t[5][1]))))
    #mAinf(V::Float64) = boltz(V,27.2,-8.7)
    
    def mA_inf (self,V):
        return boltz(V,27.2*self.mis[6][0],-8.7*self.mis[6][1])
    #taumA(V::Float64) = tauX(V,11.6,10.4,32.9,-15.2)
    
    def tau_mA (self,V):
        mis_t=self.mis_t
        return tauX(V,11.6*mis_t[6][0],10.4*mis_t[6][1],32.9*mis_t[6][2],-15.2*mis_t[6][3])
    #hAinf(V::Float64) = boltz(V,56.9,4.9)
    
    def hA_inf (self,V):
        return boltz(V,56.9*self.mis[7][0],4.9*self.mis[7][1])
    #tauhA(V::Float64) = tauX(V,38.6,29.2,38.9,-26.5)
    
    def tau_hA(self,V): 
        return tauX(V,38.6*self.mis_t[10][1],29.2*self.mis_t[10][2],38.9*self.mis_t[10][3],-26.5*self.mis_t[10][0])
    #mKCainf(V::Float64,Ca::Float64,KdCa::Float64) = (Ca/(Ca+KdCa))*(1/(1+exp((V+28.3)/-12.6)))
    
    def mK_inf(self,V,Ca,KdCa):
        return (Ca/(Ca+KdCa))*(1/(1+exp((V+28.3*self.mis[8][0])/-12.6*self.mis[8][1])))
    #taumKCa(V::Float64,tmKCa::Float64) = tmKCa*tauX(V,90.3,75.1,46.,-22.7)
    
    def tau_mK(self,V,tmKCa):
        return tmKCa*tauX(V,90.3*self.mis_t[7][0],75.1*self.mis_t[7][1],46.*self.mis_t[7][2],-22.7*self.mis_t[7][3])
    #mKdinf(V::Float64) = boltz(V,12.3,-11.8)
    
    def mKd_inf (self,V):
        return boltz(V,12.3*self.mis[9][0],-11.8*self.mis[9][1])
    #taumKd(V::Float64) = tauX(V,7.2,6.4,28.3,-19.2)
    
    def tau_mKd (self,V):
        return tauX(V,7.2*self.mis_t[8][0],6.4*self.mis_t[8][1],28.3*self.mis_t[8][2],-19.2*self.mis_t[8][3])
    #mHinf(V::Float64) = boltz(V,70.,6.)
    
    def mH_inf (self,V):
        return boltz(V,70.*self.mis[10][0],6.*self.mis[10][1])
    #taumH(V::Float64) = tauX(V,272.,-1499.,42.2,-8.73)
    
    def tau_mH(self,V):
        return tauX(V,272.*self.mis_t[9][0],-1499.*self.mis_t[9][1],42.2*self.mis_t[9][2],-8.73*self.mis_t[9][3])
    
    def mSyn_inf(self,V:np.ndarray):
        return sinf(V,-50.*self.mis[11][0],10.*self.mis[11][1])
    
mis=(np.random.rand(12,2)*0.2-0.1)*0.5+1.0
mis_t=(np.random.rand(12,4)*0.01-0.005)+1.0    
e_dyns=dyns(mis,mis_t) 
@jitclass([( "gSyn", float64[:]),("gE",float64[:]),( "mask", float64[:]),( "gamma_mask", float64[:]),( "theta", float64[:]),("mis",float64[:]),("dyns",typeof(e_dyns)),("dyns2",typeof(e_dyns))])
class neuron:
    
    min_num:float
    Iapp:float
    I1:float # Amplitude of first step input
    I2:float # Amplitude of second step input
    ti1:float # Starting time of first step input
    tf1:float # Ending time of first step input
    ti2:float # Starting time of second step input
    tf2:float # Ending time of second step input
    taunoise:float# Cutoff frequency for low-pass filtered Gaussian noise
    Ain:float# Amplitude of sinusoïdal inut
    Win:float # Frequency of  sinusoïdal inut
    gSyn:np.ndarray
    gE:np.ndarray
    gE_num:int
    syn_num:int
    VNa:float# Sodium reversal potential 45
    VCa:float # Calcium reversal potential
    VK:float # Potassium reversal potential -90
    VH:float# Reversal potential for the H-current (permeable to both sodium and potassium ions) -43
    Vleak:float # Reversal potential of leak channels
    VSyn:float
    KdCa:float
    kc:float
    gT:float # T-type calcium current maximal conductance
    gKd:float  # Delayed-rectifier potassium current maximal conductance
    gH:float# H-current maximal conductance
    gNa:float # Sodium current maximal conductance
    gA:float # A-type potassium current maximal conductance
    gKir:float  # Inward-rectifier potassium current maximal conductance
    gL:float # L-type calcium current maximal conductance
    gKCa:float # Calcium-activated potassium current maximal conductance
    gS:float
    a:float
    b:float
    C:float # Membrane capacitance
    gLeak:float # Leak current maximal conductance
    num_phi:int
    num_dinamics:int
    num_Theta:int
    pos_dinamics:int
    pos_Theta:int
    pos_phi:int
    pos_p:int
    pos_u_sys:int
    gamma:float
    mask:np.ndarray #freeze variable by setting 0 in the mask
    theta:np.ndarray 
    mis:np.ndarray
    gamma_mask:np.ndarray
    alpha:float
    tauKCa:float
    taumCa:float
    taus:float
    mismatch_dyn:bool
    mismatch_act:bool
    ob_type:str
    
    def __init__(self,p:List[float],dyns,dyns2,mismatch=False,mismatch_dyn=False,mismatch_act=False,ob_type="V"):
        
        self.min_num=1e-70 
        self.Iapp=0. # Amplitude of constant applied current
        self.I1=0. # Amplitude of first step input
        self.I2=0. # Amplitude of second step input
        self.ti1=0. # Starting time of first step input
        self.tf1=0. # Ending time of first step input
        self.ti2=0. # Starting time of second step input
        self.tf2=0. # Ending time of second step input
        self.taunoise=0.# Cutoff frequency for low-pass filtered Gaussian noise
        self.Ain=0. # Amplitude of sinusoïdal inut
        self.Win=0. # Frequency of  sinusoïdal inut
        self.syn_num=0
        self.gE_num=0
        self.VNa = 45; # Sodium reversal potential 45
        self.VCa = 120.; # Calcium reversal potential
        self.VK = -90.; # Potassium reversal potential -90
        self.VH= -43.; # Reversal potential for the H-current (permeable to both sodium and potassium ions) -43
        self.Vleak = -55.; # Reversal potential of leak channels
        self.VSyn=-120.
        self.tauKCa=1
        self.taumCa=1
        self.taus=1
        self.mismatch_dyn=mismatch_dyn
        self.mismatch_act=mismatch_act
        if mismatch:
            self.mis=(np.random.rand(6)*0.2-0.1)+1.0
        else:
            self.mis=np.ones(6)
        print(self.mis)
        
        self.gT=p[0] # T-type calcium current maximal conductance
        self.gKd=p[1]  # Delayed-rectifier potassium current maximal conductance
        self.gH=p[2] # H-current maximal conductance
        self.gNa=p[3] # Sodium current maximal conductance
        self.gA=p[4] # A-type potassium current maximal conductance
        self.gS=p[5]  # Inward-rectifier potassium current maximal conductance
        #self.gL=p[6] # L-type calcium current maximal conductance
        self.gKCa=p[6] # Calcium-activated potassium current maximal conductance
        self.C=p[7] # Membrane capacitance
        self.gLeak=p[8] # Leak current maximal conductance
        self.KdCa=p[9]
        self.kc=p[10]
        
        self.dyns=dyns
        self.dyns2=dyns2
        
        self.theta=np.array([self.gNa,self.gH,self.gT,self.gA,self.gKd,self.gLeak,self.gKCa,self.gS])
        
        self.num()
        self.ob_type=ob_type
        print(self.ob_type)
        
    def set_rev(self,p):
        self.VNa = p[0] # Sodium reversal potential 45
        self.VCa = p[1] ; # Calcium reversal potential
        self.VK = p[2] ; # Potassium reversal potential -90
        self.VH= p[3] ; # Reversal potential for the H-current (permeable to both sodium and potassium ions) -43
        self.Vleak = p[4] ; # Reversal potential of leak channels
        self.VSyn=p[5] 
        
    def num(self):
        
        num_dinamics=15+self.syn_num
        num_Theta=8+self.syn_num+self.gE_num
        self.num_phi=num_Theta
        num_p=num_Theta**2
        num_u_sys=15+self.syn_num
        print('num_Theta',num_Theta)
        
        self.num_dinamics=num_dinamics
        self.num_Theta=num_Theta
        
        self.pos_dinamics=num_dinamics
        self.pos_Theta=self.pos_dinamics+num_Theta
        self.pos_phi=self.pos_Theta+self.num_phi
        self.pos_p=self.pos_phi+num_p
        self.pos_u_sys=self.pos_p+num_u_sys
        
    def set_tau(self,tau,tau2,tau3):
        self.tauKCa=tau
        self.taumCa=tau2
        self.taus=tau3
        
    def syn_connect(self,gSyn):
        self.gSyn=gSyn #synaptic conductance[1*syn_num]
        self.syn_num=len(gSyn)
        self.num()
        
    def E_connect(self,E):
        self.gE=E
        self.gE_num=len(E)
        self.num()
        
    def set_input(self,p:List[float]):
        self.Iapp=p[0] # Amplitude of constant applied current
        self.I1=p[1] # Amplitude of first step input
        self.I2=p[2] # Amplitude of second step input
        self.ti1=p[3] # Starting time of first step input
        self.tf1=p[4] # Ending time of first step input
        self.ti2=p[5] # Starting time of second step input
        self.tf2=p[6] # Ending time of second step input
        self.taunoise=p[7] # Cutoff frequency for low-pass filtered Gaussian noise
        self.Ain=p[8] # Amplitude of sinusoïdal inut
        self.Win=p[9] # Frequency of  sinusoïdal inut

    def set_mod(self,gS,gT):
        self.gS=gS
        self.gT=gT
        

    def sys_equ(self,t,u,V_pre:np.ndarray,V_pre_E:np.ndarray):
        
        V=u[0] # Membrane potential
        mNa=u[1] # Sodium current activation
        hNa=u[2] # Sodium current inactivation
        mH=u[3] # H current activation
        mt=u[4] # T-type calcium current activation
        ht=u[5] # T-type calcium current inactivation
        mA=u[6] # A-type potassium current activation
        hA=u[7] # A-type potassium current inactivation
        mKd=u[8] # Delayed-rectifier potassium current activation
        mKCa=u[9] # L-type calcium current activation
        mS=u[10] # Intracellular calcium concentration
        hS=u[11]
        Ca=u[12]
        noise=u[13] # Input noise
        Q_Ca=u[14]
        #mSyn=[0.]
        if self.syn_num>0:
            mSyn = u[15:15+self.syn_num] # synapse
        
        
        if len(V_pre)<self.syn_num:
            raise Exception('invaild V_pre')
        
        syn_i=0.
        if self.syn_num>0:
            syn_i=-np.dot(self.gSyn,mSyn)*(V-self.VSyn)# synapse
            
        E_i=0 
        if self.gE_num>0:
            E_i=-np.dot(self.gE,(V-V_pre_E))# synapse
        
        
#         (dt)*(1/C)*(-gNa*mNa^3*hNa*(V-VNa) -gCaT*mCaT^3*hCaT*(V-VCa) -gCaS*mCaS^3*hCaS*(V-VCa)
#                     -gA*mA^3*hA*(V-VK) -gKCa*mKCa^4*(V-VK) -gKd*mKd^4*(V-VK) -gH*mH*(V-VH) 
#                     -gleak*(V-Vleak) + Iapp)
        
        du1=1/self.C*(- self.gNa*mNa**3*hNa*(V-self.VNa) - self.gH*mH*(V-self.VH) - self.gT*mt**3*ht*(V-self.VCa) 
                 - self.gA*mA**3*hA*(V-self.VK) - self.gKd*mKd**4*(V-self.VK)
                 - self.gLeak*(V-self.Vleak)
                 - self.gKCa*mKCa**4*(V-self.VK)
                 - self.gS*mS**3*hS*(V-self.VCa)
                 +syn_i
             + self.Iapp + self.I1*pulse(t,self.ti1,self.tf1) 
             + self.I2*pulse(t,self.ti2,self.tf2) + noise + self.Ain*sin(2*pi*self.Win*t))+E_i# Voltage equation
    
        du2=1/max_abs(self.dyns.tau_mNa(V),self.min_num)*(-mNa+self.dyns.mNa_inf(V)) # gating equation
        du3=1/max_abs(self.dyns.tau_hNa(V),self.min_num)*(-hNa+self.dyns.hNa_inf(V))
        du4=1/max_abs(self.dyns.tau_mH(V),self.min_num)*(-mH+self.dyns.mH_inf(V))
        du5=1/max_abs(self.dyns.tau_mt(V,self.taumCa),self.min_num)*(-mt+self.dyns.mt_inf(V))
        du6=1/max_abs(self.dyns.tau_ht(V),self.min_num)*(-ht+self.dyns.ht_inf(V))
        du7=1/max_abs(self.dyns.tau_mA(V),self.min_num)*(-mA+self.dyns.mA_inf(V))
        du8=1/max_abs(self.dyns.tau_hA(V),self.min_num)*(-hA+self.dyns.hA_inf(V))
        du9=1/max_abs(self.dyns.tau_mKd(V),self.min_num)*(-mKd+self.dyns.mKd_inf(V))
        #dmKCa= (dt)*((1/taumKCa(V,tmKCa))*(mKCainf(V,Ca,KdCa) - mKCa))
        du10=1/max_abs(self.dyns.tau_mK(V,self.tauKCa),self.min_num)*(self.dyns.mK_inf(V,Ca,self.KdCa) - mKCa)
        #dmCaS(V::Float64,mCaS::Float64,taumCa::Float64) = (dt)*((1/taumCaS(V,taumCa))*(mCaSinf(V) - mCaS))
        #dhCaS(V::Float64,hCaS::Float64) = (dt)*((1/tauhCaS(V))*(hCaSinf(V) - hCaS))
        du11=1/max_abs(self.dyns.tau_mS(V,self.taumCa),self.min_num)*(self.dyns.mS_inf(V) - mS)
        du12=1/max_abs(self.dyns.tau_hS(V),self.min_num)*(self.dyns.hS_inf(V) - hS)
        #(-kc*(gCaT*mCaT^3*hCaT*(V-VCa) +gCaS*mCaS^3*hCaS*(V-VCa)) - Ca + 0.05)
        du13=-self.kc*(self.gT*mt**3*ht*(V-self.VCa) +self.gS*mS**3*hS*(V-self.VCa)) - Ca + 0.05# Variation of intracellular calcium concentration
        du14=-noise/self.taunoise
        du15= 0 # -self.gKCa*mKCa**4*(V-self.VK) - self.gLeak*(Q_Ca)
        
        if self.syn_num>0:
            #((1/taus)*(sinf(V,-50.,10.)-s))
            du16=(1/self.taus)*(self.dyns.mSyn_inf(V_pre)-mSyn)
            result= [du1,du2,du3,du4,du5,du6,du7,du8,du9,du10,du11,du12,du13,du14,du15,*du16]
        else:
            result= [du1,du2,du3,du4,du5,du6,du7,du8,du9,du10,du11,du12,du13,du14,du15]

        return result
    
    def equ(self,t,u):
        return self.sys_equ(t,u,np.array([0.]),np.array([0.]))
    
    def set_hyp(self,gamma,alpha,variable_mask):
        #Hyperparameters
        self.gamma=gamma
        self.mask=variable_mask #freeze variable by setting 0 in the mask
        self.gamma_mask=variable_mask*gamma
        self.alpha=alpha
    
    def init_cond(self,V0):
        #Ca=(-kcvec[1]*(gCaTvec[1]*mCaT[1]^3*hCaT[1]*(V[1]-VCa) +gCaSvec[1]*mCaS[1]^3*hCaS[1]*(V[1]-VCa)) + 0.05)*ones(5)
        mS=self.dyns.mS_inf(V0)
        hS=self.dyns.hS_inf(V0)
        Ca=-self.kc*(self.gT*self.dyns.mt_inf(V0)**3*self.dyns.ht_inf(V0)*(V0-self.VCa)+ self.gS*mS**3*hS*(V0-self.VCa)) + 0.05
        
        x0 = [V0,self.dyns.mNa_inf(V0),self.dyns.hNa_inf(V0),self.dyns.mH_inf(V0),self.dyns.mt_inf(V0),self.dyns.ht_inf(V0) ,self.dyns.mA_inf(V0), self.dyns.hA_inf(V0), self.dyns.mKd_inf(V0), 
      self.dyns.mK_inf(V0,Ca,self.KdCa),mS,hS,Ca,0.,0.]
        
        for i in range(self.syn_num):
            x0.append(0)
        return x0
    
    def init_cond_2(self,V0):
        #Ca=(-kcvec[1]*(gCaTvec[1]*mCaT[1]^3*hCaT[1]*(V[1]-VCa) +gCaSvec[1]*mCaS[1]^3*hCaS[1]*(V[1]-VCa)) + 0.05)*ones(5)
        mS=self.dyns2.mS_inf(V0)
        hS=self.dyns2.hS_inf(V0)
        Ca=-self.kc*(self.gT*self.dyns2.mt_inf(V0)**3*self.dyns2.ht_inf(V0)*(V0-self.VCa)+ self.gS*mS**3*hS*(V0-self.VCa)) + 0.05
        
        x0 = [V0,self.dyns2.mNa_inf(V0),self.dyns2.hNa_inf(V0),self.dyns2.mH_inf(V0),self.dyns2.mt_inf(V0),self.dyns2.ht_inf(V0) ,self.dyns2.mA_inf(V0), self.dyns2.hA_inf(V0), self.dyns2.mKd_inf(V0), 
      self.dyns2.mK_inf(V0,Ca,self.KdCa),mS,hS,Ca,0.,0.]
        
        for i in range(self.syn_num):
            x0.append(0)
        return x0
    
    def init_cond_OB(self,V0):
        x0_=self.init_cond_2(V0)
        V1=-np.random.rand(1)*100
        x0=self.init_cond(V1[0])
        theta=np.array([*self.theta,*self.gSyn,*self.gE])
        Theta0= np.random.rand(self.num_Theta)*100.*(self.mask)+(1-self.mask)*theta
        print(Theta0)
        print(V1)
        A0=(np.ones(self.num_Theta)*0.1)
        P0=np.diag(np.diag(np.outer(A0,A0))).flatten()
        X0_=[*x0,*Theta0,*A0,*P0,*x0_]
        return X0_
    
    def init_cond_OB_mis(self,V0):
        x0_=self.init_cond_2(V0)
        V1=-np.random.rand(1)*100
        x0=self.init_cond(V1[0])
        theta=np.array([*self.theta,*self.gSyn,*self.gE])
        mis=(np.random.rand(self.num_Theta)*0.2-0.1)+1.0
        Theta0= np.random.rand(self.num_Theta)*100.*(self.mask)+(1-self.mask)*theta*mis
        print(Theta0)
        print(V1)
        A0=(np.ones(self.num_Theta)*0.1)
        P0=np.diag(np.diag(np.outer(A0,A0))).flatten()
        X0_=[*x0,*Theta0,*A0,*P0,*x0_]
        return X0_
    
    def init_cond_OB_0(self,V0):
        x0_=self.init_cond_2(V0)
        V1=-np.random.rand(1)*100
        x0=self.init_cond(V1[0])
        theta=np.array([*self.theta,*self.gSyn,*self.gE])
        mis=(np.random.rand(self.num_Theta)*0.2-0.1)+1.0
        Theta0= np.random.rand(self.num_Theta)*100.*(self.mask)
        print(Theta0)
        print(V1)
        A0=(np.ones(self.num_Theta)*0.1)
        P0=np.diag(np.diag(np.outer(A0,A0))).flatten()
        X0_=[*x0,*Theta0,*A0,*P0,*x0_]
        return X0_
    
    
    def OB_ODE_Ca(self,t,u,V_pre:np.ndarray,V_pre_E:np.ndarray):
        
        PHI8:float64[:]
        PHI9:float64[:]

        V,mNa,hNa,mH,mt,ht,mA,hA,mKd,mKCa,mS,hS,_,noise,Q_Ca=u[0:15]
    
        #mSyn=[0.]
        if self.syn_num>0:
            mSyn = u[15:15+self.syn_num] # synapse

        u_sys=u[self.pos_p:self.pos_u_sys]
        P=u[self.pos_phi:self.pos_p].reshape(self.num_phi,self.num_phi)
        Theta=u[self.pos_dinamics:self.pos_Theta]
        #print(Theta)
        phi=u[self.pos_Theta:self.pos_phi]

        obesV=u_sys[0]
        obes_Q_Ca=u_sys[12]
        Ca=u_sys[12]
    
        PHI0= -mNa**3*hNa*(obesV-self.VNa) 
        PHI1= -mH*(obesV-self.VH)
        PHI2= -mt**3*ht*(obesV-self.VCa)
        PHI3= -mA**3*hA*(obesV-self.VK)
        PHI4= - mKd**4*(obesV-self.VK)
        PHI5= -(obesV-self.Vleak)
        PHI6= -mKCa**4*(obesV-self.VK)
        PHI7= -mS**3*hS*(obesV-self.VCa)
        
        PHI_=[PHI0,PHI1,PHI2,PHI3,PHI4,PHI5,PHI6,PHI7]
        
        #PHI_I_=[0.,0.,0.,0.,0.,0.,PHI6,0.]
        
        if (self.syn_num>0) and (self.gE_num>0):
            PHI8=-mSyn*(obesV-self.VSyn)
            PHI9=-(obesV-V_pre_E)
            PHI_=[*PHI_,*PHI8,*PHI9]
            #PHI_I_=[*PHI_I_,*PHI8,*PHI9]
        else:
            if self.gE_num>0:
                PHI9=-(obesV-V_pre_E)*self.C
                PHI_=[*PHI_,*PHI9]
                #PHI_I_=[*PHI_I_,*PHI9]
            if self.syn_num>0:
                PHI8=-mSyn*(obesV-self.VSyn)
                PHI_=[*PHI_,*PHI8]
                #PHI_I_=[*PHI_I_,*PHI8]
        
        PHI=np.array(PHI_)
        #PHI_I=np.array(PHI_I_)  
        
        Current_in= self.Iapp + self.I1*pulse(t,self.ti1,self.tf1) + self.I2*pulse(t,self.ti2,self.tf2)+ self.Ain*sin(2*pi*self.Win*t)
        
        #ODEs
        du1=0#temp+temp2
        #####
        du2=1/max_abs(self.dyns2.tau_mNa(obesV),self.min_num)*(-mNa+self.dyns2.mNa_inf(obesV)) # gating equation
        du3=1/max_abs(self.dyns2.tau_hNa(obesV),self.min_num)*(-hNa+self.dyns2.hNa_inf(obesV))
        du4=1/max_abs(self.dyns2.tau_mH(obesV),self.min_num)*(-mH+self.dyns2.mH_inf(obesV))
        du5=1/max_abs(self.dyns2.tau_mt(obesV,self.taumCa),self.min_num)*(-mt+self.dyns2.mt_inf(obesV))
        du6=1/max_abs(self.dyns2.tau_ht(obesV),self.min_num)*(-ht+self.dyns2.ht_inf(obesV))
        du7=1/max_abs(self.dyns2.tau_mA(obesV),self.min_num)*(-mA+self.dyns2.mA_inf(obesV))
        du8=1/max_abs(self.dyns2.tau_hA(obesV),self.min_num)*(-hA+self.dyns2.hA_inf(obesV))
        du9=1/max_abs(self.dyns2.tau_mKd(obesV),self.min_num)*(-mKd+self.dyns2.mKd_inf(obesV))
        
        du10=1/max_abs(self.dyns2.tau_mK(obesV,self.tauKCa),self.min_num)*(self.dyns2.mK_inf(obesV,obes_Q_Ca,self.KdCa) - mKCa)
       
        du11=1/max_abs(self.dyns2.tau_mS(obesV,self.taumCa),self.min_num)*(self.dyns2.mS_inf(obesV) - mS)
        
        du12=1/max_abs(self.dyns2.tau_hS(obesV),self.min_num)*(self.dyns2.hS_inf(obesV) - hS)
        
        du13=0 # Variation of intracellular calcium concentration
        
        du14=-noise/self.taunoise
        
        du15=(-self.kc*(Theta[2]*mt**3*ht*(obesV-self.VCa) +Theta[7]*mS**3*hS*(obesV-self.VCa))- Q_Ca + 
        0.05+self.gamma*(obes_Q_Ca-Q_Ca)+self.gamma*np.dot(np.dot(phi,P),phi)*(obes_Q_Ca-Q_Ca))
        
        
        
        
        if self.syn_num>0:
            #((1/taus)*(sinf(V,-50.,10.)-s))
            du16=(1/self.taus)*(self.dyns2.mSyn_inf(V_pre)-mSyn)
            result= [du1,du2,du3,du4,du5,du6,du7,du8,du9,du10,du11,du12,du13,du14,du15,*du16]
        else:
            result= [du1,du2,du3,du4,du5,du6,du7,du8,du9,du10,du11,du12,du13,du14,du15]
        

        du17=self.gamma_mask*np.dot(P,phi)*(obes_Q_Ca-Q_Ca)

        du18=self.mask*(-self.gamma*phi+PHI)
        #du14=(np.absolute(du14)>min_num)*du14

        du19=self.alpha*P-np.dot(np.dot(P,np.outer(phi,phi)),P)

        du20=self.sys_equ(t,u_sys,V_pre,V_pre_E)


        return [*result, *du17 ,*du18,*du19.flatten(),*du20]
    
    def OB_ODE_Ca_equ(self,t,u):
        out= self.OB_ODE_Ca(t,u,np.array([0.]),np.array([0.]))

        return (out)
    
    def OB_ODE_V(self,t,u,V_pre:np.ndarray,V_pre_E:np.ndarray):
        
        PHI8:float64[:]
        PHI9:float64[:]

        V,mNa,hNa,mH,mt,ht,mA,hA,mKd,mKCa,mS,hS,Ca,noise,_=u[0:15]
    
        #mSyn=[0.]
        if self.syn_num>0:
            mSyn = u[15:15+self.syn_num] # synapse
    
        u_sys=u[self.pos_p:self.pos_u_sys]
        P=u[self.pos_phi:self.pos_p].reshape(self.num_phi,self.num_phi)
        Theta=u[self.pos_dinamics:self.pos_Theta]
        #print(Theta)
        phi=u[self.pos_Theta:self.pos_phi]

        obesV=u_sys[0]
    
        PHI0= -mNa**3*hNa*(obesV-self.VNa) 
        PHI1= -mH*(obesV-self.VH)
        PHI2= -mt**3*ht*(obesV-self.VCa)
        PHI3= -mA**3*hA*(obesV-self.VK)
        PHI4= - mKd**4*(obesV-self.VK)
        PHI5= -(obesV-self.Vleak)
        PHI6= -mKCa**4*(obesV-self.VK)
        PHI7= -mS**3*hS*(obesV-self.VCa)
        
        PHI_=[PHI0,PHI1,PHI2,PHI3,PHI4,PHI5,PHI6,PHI7]
        
        if (self.syn_num>0) and (self.gE_num>0):
            PHI8=-mSyn*(obesV-self.VSyn)
            PHI9=-(obesV-V_pre_E)
            PHI_=[*PHI_,*PHI8,*PHI9]
        else:
            if self.gE_num>0:
                PHI9=-(obesV-V_pre_E)*self.C
                PHI_=[*PHI_,*PHI9]
            if self.syn_num>0:
                PHI8=-mSyn*(obesV-self.VSyn)
                PHI_=[*PHI_,*PHI8]
        
        PHI=np.array(PHI_)
            
        
        Current_in= self.Iapp + self.I1*pulse(t,self.ti1,self.tf1) + self.I2*pulse(t,self.ti2,self.tf2)+ self.Ain*sin(2*pi*self.Win*t)


        

        #ODEs
        temp=self.gamma*(obesV-V)+self.gamma*np.dot(np.dot(phi,P),phi)*(obesV-V)# Voltage equation
        temp2=1/self.C*(np.dot(PHI,Theta) + Current_in)
        du1=temp+temp2 # V
       
        du2=1/max_abs(self.dyns2.tau_mNa(obesV),self.min_num)*(-mNa+self.dyns2.mNa_inf(obesV)) # gating equation
        du3=1/max_abs(self.dyns2.tau_hNa(obesV),self.min_num)*(-hNa+self.dyns2.hNa_inf(obesV))
        du4=1/max_abs(self.dyns2.tau_mH(obesV),self.min_num)*(-mH+self.dyns2.mH_inf(obesV))
        du5=1/max_abs(self.dyns2.tau_mt(obesV,self.taumCa),self.min_num)*(-mt+self.dyns2.mt_inf(obesV))
        du6=1/max_abs(self.dyns2.tau_ht(obesV),self.min_num)*(-ht+self.dyns2.ht_inf(obesV))
        du7=1/max_abs(self.dyns2.tau_mA(obesV),self.min_num)*(-mA+self.dyns2.mA_inf(obesV))
        du8=1/max_abs(self.dyns2.tau_hA(obesV),self.min_num)*(-hA+self.dyns2.hA_inf(obesV))
        du9=1/max_abs(self.dyns2.tau_mKd(obesV),self.min_num)*(-mKd+self.dyns2.mKd_inf(obesV))
        du10=1/max_abs(self.dyns2.tau_mK(obesV,self.tauKCa),self.min_num)*(self.dyns2.mK_inf(obesV,Ca,self.KdCa) - mKCa)
        du11=1/max_abs(self.dyns2.tau_mS(obesV,self.taumCa),self.min_num)*(self.dyns2.mS_inf(obesV) - mS)
        du12=1/max_abs(self.dyns2.tau_hS(obesV),self.min_num)*(self.dyns2.hS_inf(obesV) - hS)
        du13=-self.kc*(Theta[2]*mt**3*ht*(obesV-self.VCa) +Theta[7]*mS**3*hS*(obesV-self.VCa)) - Ca + 0.05# Variation of intracellular calcium concentration
        
        du14=-noise/self.taunoise
        du15=0
        
        if self.syn_num>0:
            #((1/taus)*(sinf(V,-50.,10.)-s))
            du16=(1/self.taus)*(self.dyns2.mSyn_inf(V_pre)-mSyn)
            result= [du1,du2,du3,du4,du5,du6,du7,du8,du9,du10,du11,du12,du13,du14,du15,*du16]
        else:
            result= [du1,du2,du3,du4,du5,du6,du7,du8,du9,du10,du11,du12,du13,du14,du15]
        

        du17=self.gamma_mask*np.dot(P,phi)*(obesV-V)

        du18=self.mask*(-self.gamma*phi+PHI)
        #du14=(np.absolute(du14)>min_num)*du14

        du19=self.alpha*P-np.dot(np.dot(P,np.outer(phi,phi)),P)

        du20=self.sys_equ(t,u_sys,V_pre,V_pre_E)


        return [*result, *du17 ,*du18,*du19.flatten(),*du20]
    
    def OB_ODE_V_equ(self,t,u):
        out= self.OB_ODE_V(t,u,np.array([0.]),np.array([0.]))

        return (out)
    
    def OB_ODE_V_Ca(self,t,u,V_pre:np.ndarray,V_pre_E:np.ndarray):
        
        PHI8:float64[:]
        PHI9:float64[:]

        V,mNa,hNa,mH,mt,ht,mA,hA,mKd,mKCa,mS,hS,_,noise,_=u[0:15]
    
        #mSyn=[0.]
        if self.syn_num>0:
            mSyn = u[15:15+self.syn_num] # synapse

        u_sys=u[self.pos_p:self.pos_u_sys]
        P=u[self.pos_phi:self.pos_p].reshape(self.num_phi,self.num_phi)
        Theta=u[self.pos_dinamics:self.pos_Theta]
        #print(Theta)
        phi=u[self.pos_Theta:self.pos_phi]

        obesV=u_sys[0]
        Ca=u_sys[12]
    
        PHI0= -mNa**3*hNa*(obesV-self.VNa) 
        PHI1= -mH*(obesV-self.VH)
        PHI2= -mt**3*ht*(obesV-self.VCa)
        PHI3= -mA**3*hA*(obesV-self.VK)
        PHI4= - mKd**4*(obesV-self.VK)
        PHI5= -(obesV-self.Vleak)
        PHI6= -mKCa**4*(obesV-self.VK)
        PHI7= -mS**3*hS*(obesV-self.VCa)
        
        PHI_=[PHI0,PHI1,PHI2,PHI3,PHI4,PHI5,PHI6,PHI7]
        
        if (self.syn_num>0) and (self.gE_num>0):
            PHI8=-mSyn*(obesV-self.VSyn)
            PHI9=-(obesV-V_pre_E)
            PHI_=[*PHI_,*PHI8,*PHI9]
        else:
            if self.gE_num>0:
                PHI9=-(obesV-V_pre_E)*self.C
                PHI_=[*PHI_,*PHI9]
            if self.syn_num>0:
                PHI8=-mSyn*(obesV-self.VSyn)
                PHI_=[*PHI_,*PHI8]
        
        PHI=np.array(PHI_)
            
        
        Current_in= self.Iapp + self.I1*pulse(t,self.ti1,self.tf1) + self.I2*pulse(t,self.ti2,self.tf2)+ self.Ain*sin(2*pi*self.Win*t)


        

        #ODEs
        temp=self.gamma*(obesV-V)+self.gamma*np.dot(np.dot(phi,P),phi)*(obesV-V)# Voltage equation
        temp2=1/self.C*(np.dot(PHI,Theta) + Current_in)
        du1=temp+temp2 # V
       
        du2=1/max_abs(self.dyns2.tau_mNa(obesV),self.min_num)*(-mNa+self.dyns2.mNa_inf(obesV)) # gating equation
        du3=1/max_abs(self.dyns2.tau_hNa(obesV),self.min_num)*(-hNa+self.dyns2.hNa_inf(obesV))
        du4=1/max_abs(self.dyns2.tau_mH(obesV),self.min_num)*(-mH+self.dyns2.mH_inf(obesV))
        du5=1/max_abs(self.dyns2.tau_mt(obesV,self.taumCa),self.min_num)*(-mt+self.dyns2.mt_inf(obesV))
        du6=1/max_abs(self.dyns2.tau_ht(obesV),self.min_num)*(-ht+self.dyns2.ht_inf(obesV))
        du7=1/max_abs(self.dyns2.tau_mA(obesV),self.min_num)*(-mA+self.dyns2.mA_inf(obesV))
        du8=1/max_abs(self.dyns2.tau_hA(obesV),self.min_num)*(-hA+self.dyns2.hA_inf(obesV))
        du9=1/max_abs(self.dyns2.tau_mKd(obesV),self.min_num)*(-mKd+self.dyns2.mKd_inf(obesV))
        du10=1/max_abs(self.dyns2.tau_mK(obesV,self.tauKCa),self.min_num)*(self.dyns2.mK_inf(obesV,Ca,self.KdCa) - mKCa)
        du11=1/max_abs(self.dyns2.tau_mS(obesV,self.taumCa),self.min_num)*(self.dyns2.mS_inf(obesV) - mS)
        du12=1/max_abs(self.dyns2.tau_hS(obesV),self.min_num)*(self.dyns2.hS_inf(obesV) - hS)
        du13=0 #calcium concentration (not using this since we observe Ca directly)
        du14=-noise/self.taunoise
        
        
        du15=0
        
        if self.syn_num>0:
            #((1/taus)*(sinf(V,-50.,10.)-s))
            du16=(1/self.taus)*(self.dyns2.mSyn_inf(V_pre)-mSyn)
            result= [du1,du2,du3,du4,du5,du6,du7,du8,du9,du10,du11,du12,du13,du14,du15,*du16]
        else:
            result= [du1,du2,du3,du4,du5,du6,du7,du8,du9,du10,du11,du12,du13,du14,du15]
        

        du17=self.gamma_mask*np.dot(P,phi)*(obesV-V)

        du18=self.mask*(-self.gamma*phi+PHI)
        #du14=(np.absolute(du14)>min_num)*du14

        du19=self.alpha*P-np.dot(np.dot(P,np.outer(phi,phi)),P)

        du20=self.sys_equ(t,u_sys,V_pre,V_pre_E)


        return [*result, *du17 ,*du18,*du19.flatten(),*du20]
    
    def OB_ODE_V_Ca_equ(self,t,u):
        out= self.OB_ODE_V_Ca(t,u,np.array([0.]),np.array([0.]))

        return (out)
    
    def OB_ODE(self,t,u,V_pre:np.ndarray,V_pre_E:np.ndarray):
        if self.ob_type=="V_Ca":
            return self.OB_ODE_V_Ca(t,u,V_pre,V_pre_E)
        elif self.ob_type=="Ca":
            return self.OB_ODE_Ca(t,u,V_pre,V_pre_E)
        elif self.ob_type=="V":
            return self.OB_ODE_V(t,u,V_pre,V_pre_E)
        else:
            print("wrong observer type")
            raise
    
@jitclass([( "gSyn", float64[:]),("gE",float64[:]),( "mask", float64[:]),( "gamma_mask", float64[:]),( "theta", float64[:])])
class neuron_diag:
    
    min_num:float
    Iapp:float
    I1:float # Amplitude of first step input
    I2:float # Amplitude of second step input
    ti1:float # Starting time of first step input
    tf1:float # Ending time of first step input
    ti2:float # Starting time of second step input
    tf2:float # Ending time of second step input
    taunoise:float# Cutoff frequency for low-pass filtered Gaussian noise
    Ain:float# Amplitude of sinusoïdal inut
    Win:float # Frequency of  sinusoïdal inut
    gSyn:np.ndarray
    gE:np.ndarray
    gE_num:int
    syn_num:int
    VNa:float# Sodium reversal potential 45
    VCa:float # Calcium reversal potential
    VK:float # Potassium reversal potential -90
    VH:float# Reversal potential for the H-current (permeable to both sodium and potassium ions) -43
    Vleak:float # Reversal potential of leak channels
    VSyn:float
    KdCa:float
    kc:float
    gT:float # T-type calcium current maximal conductance
    gKd:float  # Delayed-rectifier potassium current maximal conductance
    gH:float# H-current maximal conductance
    gNa:float # Sodium current maximal conductance
    gA:float # A-type potassium current maximal conductance
    gKir:float  # Inward-rectifier potassium current maximal conductance
    gL:float # L-type calcium current maximal conductance
    gKCa:float # Calcium-activated potassium current maximal conductance
    gS:float
    a:float
    b:float
    C:float # Membrane capacitance
    gLeak:float # Leak current maximal conductance
    num_phi:int
    num_dinamics:int
    num_Theta:int
    pos_dinamics:int
    pos_Theta:int
    pos_phi:int
    pos_p:int
    pos_u_sys:int
    gamma:float
    mask:np.ndarray #freeze variable by setting 0 in the mask
    theta:np.ndarray 
    gamma_mask:np.ndarray
    alpha:float
    tauKCa:float
    taumCa:float
    taus:float
    
    def __init__(self,p:List[float]):
        
        self.min_num=1e-70 
        self.Iapp=0. # Amplitude of constant applied current
        self.I1=0. # Amplitude of first step input
        self.I2=0. # Amplitude of second step input
        self.ti1=0. # Starting time of first step input
        self.tf1=0. # Ending time of first step input
        self.ti2=0. # Starting time of second step input
        self.tf2=0. # Ending time of second step input
        self.taunoise=0.# Cutoff frequency for low-pass filtered Gaussian noise
        self.Ain=0. # Amplitude of sinusoïdal inut
        self.Win=0. # Frequency of  sinusoïdal inut
        self.syn_num=0
        self.gE_num=0
        self.VNa = 45; # Sodium reversal potential 45
        self.VCa = 120.; # Calcium reversal potential
        self.VK = -90.; # Potassium reversal potential -90
        self.VH= -43.; # Reversal potential for the H-current (permeable to both sodium and potassium ions) -43
        self.Vleak = -55.; # Reversal potential of leak channels
        self.VSyn=-120.
        self.tauKCa=1
        self.taumCa=1
        self.taus=1
        
        self.gT=p[0] # T-type calcium current maximal conductance
        self.gKd=p[1]  # Delayed-rectifier potassium current maximal conductance
        self.gH=p[2] # H-current maximal conductance
        self.gNa=p[3] # Sodium current maximal conductance
        self.gA=p[4] # A-type potassium current maximal conductance
        self.gS=p[5]  # Inward-rectifier potassium current maximal conductance
        #self.gL=p[6] # L-type calcium current maximal conductance
        self.gKCa=p[6] # Calcium-activated potassium current maximal conductance
        self.C=p[7] # Membrane capacitance
        self.gLeak=p[8] # Leak current maximal conductance
        self.KdCa=p[9]
        self.kc=p[10]
        
#         PHI0= -mNa**3*hNa*(obesV-self.VNa) 
#         PHI1= -mH*(obesV-self.VH)
#         PHI2= -mt**3*ht*(obesV-self.VCa)
#         PHI3= -mA**3*hA*(obesV-self.VK)
#         PHI4= - mKd**4*(obesV-self.VK)
#         PHI5= -(obesV-self.Vleak)
#         PHI6= -mKCa**4*(obesV-self.VK)
#         PHI7= -mS**3*hS*(obesV-self.VCa)
        
        self.theta=np.array([self.gNa,self.gH,self.gT,self.gA,self.gKd,self.gLeak,self.gKCa,self.gS])
        
        self.num()
        
    def set_rev(self,p):
        self.VNa = p[0] # Sodium reversal potential 45
        self.VCa = p[1] ; # Calcium reversal potential
        self.VK = p[2] ; # Potassium reversal potential -90
        self.VH= p[3] ; # Reversal potential for the H-current (permeable to both sodium and potassium ions) -43
        self.Vleak = p[4] ; # Reversal potential of leak channels
        self.VSyn=p[5] 
        
    def num(self):
        
        num_dinamics=15+self.syn_num
        num_Theta=8+self.syn_num+self.gE_num
        self.num_phi=num_Theta
        num_p=num_Theta
        num_u_sys=15+self.syn_num
        print('num_Theta',num_Theta)
        
        self.num_dinamics=num_dinamics
        self.num_Theta=num_Theta
        
        self.pos_dinamics=num_dinamics
        self.pos_Theta=self.pos_dinamics+num_Theta
        self.pos_phi=self.pos_Theta+self.num_phi
        self.pos_p=self.pos_phi+num_p
        self.pos_u_sys=self.pos_p+num_u_sys
        
    def set_tau(self,tau,tau2,tau3):
        self.tauKCa=tau
        self.taumCa=tau2
        self.taus=tau3
        
    def syn_connect(self,gSyn):
        self.gSyn=gSyn #synaptic conductance[1*syn_num]
        self.syn_num=len(gSyn)
        self.num()
        
    def E_connect(self,E):
        self.gE=E
        self.gE_num=len(E)
        self.num()
        
    def set_input(self,p:List[float]):
        self.Iapp=p[0] # Amplitude of constant applied current
        self.I1=p[1] # Amplitude of first step input
        self.I2=p[2] # Amplitude of second step input
        self.ti1=p[3] # Starting time of first step input
        self.tf1=p[4] # Ending time of first step input
        self.ti2=p[5] # Starting time of second step input
        self.tf2=p[6] # Ending time of second step input
        self.taunoise=p[7] # Cutoff frequency for low-pass filtered Gaussian noise
        self.Ain=p[8] # Amplitude of sinusoïdal inut
        self.Win=p[9] # Frequency of  sinusoïdal inut

    def set_mod(self,gS,gT):
        self.gS=gS
        self.gT=gT
        

    def sys_equ(self,t,u,V_pre:np.ndarray,V_pre_E:np.ndarray):
        
        V=u[0] # Membrane potential
        mNa=u[1] # Sodium current activation
        hNa=u[2] # Sodium current inactivation
        mH=u[3] # H current activation
        mt=u[4] # T-type calcium current activation
        ht=u[5] # T-type calcium current inactivation
        mA=u[6] # A-type potassium current activation
        hA=u[7] # A-type potassium current inactivation
        mKd=u[8] # Delayed-rectifier potassium current activation
        mKCa=u[9] # L-type calcium current activation
        mS=u[10] # Intracellular calcium concentration
        hS=u[11]
        Ca=u[12]
        noise=u[13] # Input noise
        Q_Ca=u[14]
        #mSyn=[0.]
        if self.syn_num>0:
            mSyn = u[15:15+self.syn_num] # synapse
        
        
        if len(V_pre)<self.syn_num:
            raise Exception('invaild V_pre')
        
        syn_i=0.
        if self.syn_num>0:
            syn_i=-np.dot(self.gSyn,mSyn)*(V-self.VSyn)# synapse
            
        E_i=0 
        if self.gE_num>0:
            E_i=-np.dot(self.gE,(V-V_pre_E))# synapse
        
        
#         (dt)*(1/C)*(-gNa*mNa^3*hNa*(V-VNa) -gCaT*mCaT^3*hCaT*(V-VCa) -gCaS*mCaS^3*hCaS*(V-VCa)
#                     -gA*mA^3*hA*(V-VK) -gKCa*mKCa^4*(V-VK) -gKd*mKd^4*(V-VK) -gH*mH*(V-VH) 
#                     -gleak*(V-Vleak) + Iapp)
        
        du1=1/self.C*(- self.gNa*mNa**3*hNa*(V-self.VNa) - self.gH*mH*(V-self.VH) - self.gT*mt**3*ht*(V-self.VCa) 
                 - self.gA*mA**3*hA*(V-self.VK) - self.gKd*mKd**4*(V-self.VK)
                 - self.gLeak*(V-self.Vleak)
                 - self.gKCa*mKCa**4*(V-self.VK)
                 - self.gS*mS**3*hS*(V-self.VCa)
                 +syn_i
             + self.Iapp + self.I1*pulse(t,self.ti1,self.tf1) 
             + self.I2*pulse(t,self.ti2,self.tf2) + noise + self.Ain*sin(2*pi*self.Win*t))+E_i# Voltage equation
    
        du2=1/max_abs(tau_mNa(V),self.min_num)*(-mNa+mNa_inf(V)) # gating equation
        du3=1/max_abs(tau_hNa(V),self.min_num)*(-hNa+hNa_inf(V))
        du4=1/max_abs(tau_mH(V),self.min_num)*(-mH+mH_inf(V))
        du5=1/max_abs(tau_mt(V,self.taumCa),self.min_num)*(-mt+mt_inf(V))
        du6=1/max_abs(tau_ht(V),self.min_num)*(-ht+ht_inf(V))
        du7=1/max_abs(tau_mA(V),self.min_num)*(-mA+mA_inf(V))
        du8=1/max_abs(tau_hA(V),self.min_num)*(-hA+hA_inf(V))
        du9=1/max_abs(tau_mKd(V),self.min_num)*(-mKd+mKd_inf(V))
        #dmKCa= (dt)*((1/taumKCa(V,tmKCa))*(mKCainf(V,Ca,KdCa) - mKCa))
        du10=1/max_abs(tau_mK(V,self.tauKCa),self.min_num)*(mK_inf(V,Ca,self.KdCa) - mKCa)
        #dmCaS(V::Float64,mCaS::Float64,taumCa::Float64) = (dt)*((1/taumCaS(V,taumCa))*(mCaSinf(V) - mCaS))
        #dhCaS(V::Float64,hCaS::Float64) = (dt)*((1/tauhCaS(V))*(hCaSinf(V) - hCaS))
        du11=1/max_abs(tau_mS(V,self.taumCa),self.min_num)*(mS_inf(V) - mS)
        du12=1/max_abs(tau_hS(V),self.min_num)*(hS_inf(V) - hS)
        #(-kc*(gCaT*mCaT^3*hCaT*(V-VCa) +gCaS*mCaS^3*hCaS*(V-VCa)) - Ca + 0.05)
        du13=-self.kc*(self.gT*mt**3*ht*(V-self.VCa) +self.gS*mS**3*hS*(V-self.VCa)) - Ca + 0.05# Variation of intracellular calcium concentration
        du14=-noise/self.taunoise
        du15= -self.gKCa*mKCa**4*(V-self.VK) - self.gLeak*(Q_Ca)
        
        if self.syn_num>0:
            #((1/taus)*(sinf(V,-50.,10.)-s))
            du16=(1/self.taus)*(sinf(V_pre,-50.,10.)-mSyn)
            result= [du1,du2,du3,du4,du5,du6,du7,du8,du9,du10,du11,du12,du13,du14,du15,*du16]
        else:
            result= [du1,du2,du3,du4,du5,du6,du7,du8,du9,du10,du11,du12,du13,du14,du15]

        return result
    
    def equ(self,t,u):
        return self.sys_equ(t,u,np.array([0.]),np.array([0.]))
    
    def set_hyp(self,gamma,alpha,variable_mask):
        #Hyperparameters
        self.gamma=gamma
        self.mask=variable_mask #freeze variable by setting 0 in the mask
        self.gamma_mask=variable_mask*gamma
        self.alpha=alpha
    
    def init_cond(self,V0):
        #Ca=(-kcvec[1]*(gCaTvec[1]*mCaT[1]^3*hCaT[1]*(V[1]-VCa) +gCaSvec[1]*mCaS[1]^3*hCaS[1]*(V[1]-VCa)) + 0.05)*ones(5)
        mS=mS_inf(V0)
        hS=hS_inf(V0)
        Ca=-self.kc*(self.gT*mt_inf(V0)**3*ht_inf(V0)*(V0-self.VCa)+ self.gS*mS**3*hS*(V0-self.VCa)) + 0.05
        
        x0 = [V0,mNa_inf(V0),hNa_inf(V0),mH_inf(V0),mt_inf(V0),ht_inf(V0) ,mA_inf(V0), hA_inf(V0), mKd_inf(V0), 
      mK_inf(V0,Ca,self.KdCa),mS,hS,Ca,0.,0.]
        
        for i in range(self.syn_num):
            x0.append(0)
        return x0
    
    def init_cond_OB(self,V0):
        x0_=self.init_cond(V0)
        V1=-np.random.rand(1)*60
        x0=self.init_cond(V1[0])
        theta=np.array([*self.theta,*self.gSyn,*self.gE])
        Theta0= np.random.rand(self.num_Theta)*1.*(self.mask)+theta
        print(Theta0)
        print(V1)
        A0=(np.ones(self.num_Theta)*0.1)
        P0=np.ones(self.num_Theta)
        X0_=[*x0,*Theta0,*A0,*P0,*x0_]
        return X0_
    
    def OB_ODE(self,t,u,V_pre:np.ndarray,V_pre_E:np.ndarray):
        
        PHI8:float64[:]
        PHI9:float64[:]

        V,mNa,hNa,mH,mt,ht,mA,hA,mKd,mKCa,mS,hS,Ca,noise,Q_Ca=u[0:15]
    
        #mSyn=[0.]
        if self.syn_num>0:
            mSyn = u[15:15+self.syn_num] # synapse

        u_sys=u[self.pos_p:self.pos_u_sys]
        P=u[self.pos_phi:self.pos_p]
        Theta=u[self.pos_dinamics:self.pos_Theta]
        #print(Theta)
        phi=u[self.pos_Theta:self.pos_phi]

        obesV=u_sys[0]
    
        PHI0= -mNa**3*hNa*(obesV-self.VNa) 
        PHI1= -mH*(obesV-self.VH)
        PHI2= -mt**3*ht*(obesV-self.VCa)
        PHI3= -mA**3*hA*(obesV-self.VK)
        PHI4= - mKd**4*(obesV-self.VK)
        PHI5= -(obesV-self.Vleak)
        PHI6= -mKCa**4*(obesV-self.VK)
        PHI7= -mS**3*hS*(obesV-self.VCa)
        
        PHI_=[PHI0,PHI1,PHI2,PHI3,PHI4,PHI5,PHI6,PHI7]
        
        #PHI_I_=[0.,0.,0.,0.,0.,0.,PHI6,0.]
        
        if (self.syn_num>0) and (self.gE_num>0):
            PHI8=-mSyn*(obesV-self.VSyn*1.03)
            PHI9=-(obesV-V_pre_E)
            PHI_=[*PHI_,*PHI8,*PHI9]
            #PHI_I_=[*PHI_I_,*PHI8,*PHI9]
        else:
            if self.gE_num>0:
                PHI9=-(obesV-V_pre_E)*self.C
                PHI_=[*PHI_,*PHI9]
                #PHI_I_=[*PHI_I_,*PHI9]
            if self.syn_num>0:
                PHI8=-mSyn*(obesV-self.VSyn)
                PHI_=[*PHI_,*PHI8]
                #PHI_I_=[*PHI_I_,*PHI8]
        
        PHI=np.array(PHI_)
        #PHI_I=np.array(PHI_I_)  
        
        Current_in= self.Iapp + self.I1*pulse(t,self.ti1,self.tf1) + self.I2*pulse(t,self.ti2,self.tf2)+ self.Ain*sin(2*pi*self.Win*t)


        
        obes_Q_Ca=u_sys[12]
        #ODEs
        # Voltage equation
        #temp=self.gamma*(obesV-V)+self.gamma*np.dot(np.dot(phi,P),phi)*(obesV-V)# Voltage equation
        #temp2=1/self.C*(np.dot(PHI,Theta) + Current_in)
        du1=0#temp+temp2
        #####
        du2=1/max_abs(tau_mNa(obesV),self.min_num)*(-mNa+mNa_inf(obesV)) # gating equation
        du3=1/max_abs(tau_hNa(obesV),self.min_num)*(-hNa+hNa_inf(obesV))
        du4=1/max_abs(tau_mH(obesV),self.min_num)*(-mH+mH_inf(obesV))
        du5=1/max_abs(tau_mt(obesV,self.taumCa),self.min_num)*(-mt+mt_inf(obesV))
        du6=1/max_abs(tau_ht(obesV),self.min_num)*(-ht+ht_inf(obesV))
        du7=1/max_abs(tau_mA(obesV),self.min_num)*(-mA+mA_inf(obesV))
        du8=1/max_abs(tau_hA(obesV),self.min_num)*(-hA+hA_inf(obesV))
        du9=1/max_abs(tau_mKd(obesV),self.min_num)*(-mKd+mKd_inf(obesV))
        
        du10=1/max_abs(tau_mK(obesV,self.tauKCa),self.min_num)*(mK_inf(obesV,Ca,self.KdCa) - mKCa)
        #dmCaS(V::Float64,mCaS::Float64,taumCa::Float64) = (dt)*((1/taumCaS(V,taumCa))*(mCaSinf(V) - mCaS))
        #dhCaS(V::Float64,hCaS::Float64) = (dt)*((1/tauhCaS(V))*(hCaSinf(V) - hCaS))
        du11=1/max_abs(tau_mS(obesV,self.taumCa),self.min_num)*(mS_inf(obesV) - mS)
        du12=1/max_abs(tau_hS(obesV),self.min_num)*(hS_inf(obesV) - hS)
        
        #(-kc*(gCaT*mCaT^3*hCaT*(V-VCa) +gCaS*mCaS^3*hCaS*(V-VCa)) - Ca + 0.05)
        du13=-self.kc*(self.gT*mt**3*ht*(obesV-self.VCa) +self.gS*mS**3*hS*(obesV-self.VCa)) - Ca + 0.05# Variation of intracellular calcium concentration
        
        du14=-noise/self.taunoise
        
        du15=(-self.kc*(Theta[2]*mt**3*ht*(obesV-self.VCa) +Theta[7]*mS**3*hS*(obesV-self.VCa))- Q_Ca + 
        0.05+self.gamma*(obes_Q_Ca-Q_Ca)+self.gamma*np.dot(np.square(phi),P)*(obes_Q_Ca-Q_Ca))
        
        
        
        
        if self.syn_num>0:
            #((1/taus)*(sinf(V,-50.,10.)-s))
            du16=(1/self.taus)*(sinf(V_pre,-50.,10.)-mSyn)
            result= [du1,du2,du3,du4,du5,du6,du7,du8,du9,du10,du11,du12,du13,du14,du15,*du16]
        else:
            result= [du1,du2,du3,du4,du5,du6,du7,du8,du9,du10,du11,du12,du13,du14,du15]
        

        du17=self.gamma_mask*np.multiply(P,phi)*(obes_Q_Ca-Q_Ca)

        du18=self.mask*(-self.gamma*phi+PHI)
        #du14=(np.absolute(du14)>min_num)*du14

        du19=self.alpha*P-np.multiply(np.square(P),np.square(phi))

        du20=self.sys_equ(t,u_sys,V_pre,V_pre_E)


        return [*result, *du17 ,*du18,*du19.flatten(),*du20]
    
    def OB_ODE_equ(self,t,u):
        out= self.OB_ODE(t,u,np.array([0.]),np.array([0.]))

        return (out)
import copy
class network:
    
    def __init__(self,cells,connections,E_connections,learn_topo=False):
        self.mod=False
        self.cells=cells.copy()
        self.connections=connections
        self.E_connections=E_connections
        self.num_cell=len(cells)
        self.links=[[] for i in range(self.num_cell)]
        self.links_E=[[] for i in range(self.num_cell)]
        self.links_ob=[[] for i in range(self.num_cell)]
        self.links_E_ob=[[] for i in range(self.num_cell)]
        
        pos=[0 for i in range(self.num_cell)]
        pos_ob=[0 for i in range(self.num_cell)]
        
        
        if (learn_topo==False):

            for i in range(self.num_cell):   
                strength=[]
                E_strength=[]
                for j in range(self.num_cell):
                    if connections[i][j]!=0:
                        strength.append(connections[i][j])
                    if E_connections[i][j]!=0:
                        E_strength.append(E_connections[i][j])
                        
                self.cells[i].E_connect(np.array(E_strength))     
                self.cells[i].syn_connect(np.array(strength))  
                
            for i in range(self.num_cell-1):
                pos[i+1]=pos[i]+cells[i].pos_dinamics
                pos_ob[i+1]=pos_ob[i]+cells[i].pos_u_sys 
                
            for i in range(self.num_cell):   
                for j in range(self.num_cell):
                    if connections[i][j]!=0:
                        self.links[i].append(pos[j])
                        self.links_ob[i].append(pos_ob[j]+cells[j].pos_p)
                    if E_connections[i][j]!=0:
                        self.links_E[i].append(pos[j])
                        self.links_E_ob[i].append(pos_ob[j]+cells[j].pos_p)
                           
        else:
            cells[0].syn_connect(np.array(connections[0]))
            cells[0].E_connect(np.array(E_connections[0]))
            for i in range(self.num_cell-1):
                pos[i+1]=pos[i]+cells[i].pos_dinamics
                pos_ob[i+1]=pos_ob[i]+cells[i].pos_u_sys
                self.cells[i+1].syn_connect(np.array(connections[i+1]))
                self.cells[i+1].E_connect(np.array(E_connections[i+1]))   
                
            for i in range(self.num_cell):   
                for j in range(self.num_cell):
                    self.links[i].append(pos[j])
                    self.links_ob[i].append(pos_ob[j]+cells[j].pos_p)
                    self.links_E[i].append(pos[j])
                    self.links_E_ob[i].append(pos_ob[j]+cells[j].pos_p)
                    
        self.pos_ob=pos_ob
        self.pos=pos
       
    def set_mod(self,fgS,fgT):
        self.mod=True
        self.fgS=fgS
        self.fgT=fgT
        
    
    def sys_equ(self,t,u):
        equ=[]
        count=0
        for i in range(self.num_cell):
            if self.mod :
                self.cells[i].set_mod(self.fgS(i,t),self.fgT(i,t))
            equ=[*equ,*self.cells[i].sys_equ(t
                                        ,np.array(u[count:count+self.cells[i].pos_dinamics])
                                        ,np.array(u[self.links[i]])
                                        ,np.array(u[self.links_E[i]])
                                       )]
            count+=self.cells[i].pos_dinamics
        return equ
    
    def ob_equ(self,t,u):
        equ=[]
        count=0
        for i in range(self.num_cell):
            if self.mod :
                self.cells[i].set_mod(self.fgS(i,t),self.fgT(i,t))
            equ=[*equ,*self.cells[i].OB_ODE(t
                                        ,np.array(u[count:count+self.cells[i].pos_u_sys])
                                        ,np.array(u[self.links_ob[i]])
                                        ,np.array(u[self.links_E_ob[i]])
                                       )]
            count+=self.cells[i].pos_u_sys
        return equ

