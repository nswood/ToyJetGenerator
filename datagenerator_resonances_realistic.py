import numpy as np
import multiprocessing as mp
from pylorentz import Momentum4
from scipy.stats import beta
import multiprocessing as mp
from multiprocessing import Process, Pool
from dataclasses import dataclass
import math
import copy

@dataclass
class particle:
    mom: Momentum4
    randtheta: float
    z: float
    m1: float
    m2: float
    prong_label: int
    part_label: int
    part_parent_label: int
    resonance_origin: str  # Label for resonance origin


class jet_data_generator(object):
    """
    Input takes the following form.

    massprior: "signal" or "background" (signal to use Gaussian prior, background for uniform)
    nprong: number of particles after hard splitting
    nparticle: total number of particles after showering

    resonance_data: List of dictionaries, each defining a resonance:
        {
            'mass': float,  # Mass of the resonance
            'relative_ratio': float,  # Relative ratio of occurrence
            'decay_products': int  # Number of decay products
        }

    total_resonance_prob: float  # Total probability of any resonance occurring

    """
    def __init__(self, 
                 massprior, 
                 nprong, 
                 nparticle, 
                 doFixP, 
                 resonance_data, 
                 total_resonance_prob,
                 max_resonance_per_jet=3, 
                 doMultiprocess=False, ncore=0):
        super(jet_data_generator, self).__init__()
        self.massprior = massprior
        self.nprong = nprong
        self.nparticle = nparticle
        self.zsoft = []
        self.zhard = []
        self.z = []
        self.randtheta = []
        self.doFixP = doFixP
        self.doMultiprocess = doMultiprocess
        self.ncore = ncore
        

        # Normalize resonance probabilities
        total_ratio = sum(r['relative_ratio'] for r in resonance_data)
        for r in resonance_data:
            r['probability'] = (r['relative_ratio'] / total_ratio) * total_resonance_prob

        self.resonance_data = resonance_data  # Store resonance configurations
        self.max_resonance_per_jet = max_resonance_per_jet
    def reverse_insort(self, a, x, lo=0, hi=None):
        """Insert item x in list a, and keep it reverse-sorted assuming a
        is reverse-sorted. The key compared is the invariant mass of the 4-vector

        If x is already in a, insert it to the right of the rightmost x.

        Optional args lo (default 0) and hi (default len(a)) bound the
        slice of a to be searched.
        """
        if lo < 0:
            raise ValueError('lo must be non-negative')
        if hi is None:
            hi = len(a)
        while lo < hi:
            mid = (lo+hi)//2
            if (x.mom.m > a[mid].mom.m and x.mom.p > 1) or  (x.mom.p >  a[mid].mom.p and x.mom.p < 1): hi = mid
            else: lo = mid+1
        a.insert(lo, x)
        return lo

    def rotation_matrix(self,axis, theta):
        """
        Return the rotation matrix associated with counterclockwise rotation about
        the given axis by theta radians.
        """
        axis = np.asarray(axis)
        axis = axis / math.sqrt(np.dot(axis, axis))
        a = math.cos(theta / 2.0)
        b, c, d = -axis * math.sin(theta / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                        [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                        [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

        
    def theta_to_eta(self,theta):
        if theta > np.pi:
            theta = 2*np.pi - theta
        return -np.log(np.tan(theta/2))

    def sintheta1(self,z,theta):
        return (-z/(1-z))*np.sin(theta)

    def mass(self,z,theta):
        sint1=self.sintheta1(z,theta)
        cost1=np.sqrt(1-sint1**2)
        p=z*np.cos(theta)+(1-z)*cost1
        return np.sqrt(1-p**2)

    def gamma(self,z,theta):
        return 1./self.mass(z,theta)

    def betaval(self,gamma):
        return np.sqrt(1-1/gamma**2)

    def sinthetaR(self,z,theta):
        gammavar=self.gamma(z,theta)
        K=gammavar*np.tan(theta)
        betavar=self.betaval(gammavar)
        return 2*K/(K**2+betavar)

    def sinthetaR2(self,z,theta):#more robust solution
        mom=(self.mass(z,theta)/2)
        return z*np.sin(theta)/mom
    
    def restmom(self,z,theta):
        sintR=self.sinthetaR(z,theta)
        gammavar=self.gamma(z,theta)
        betavar=self.betaval(gammavar)
        return z*betavar*np.sin(theta)/sintR

    def randmtheta(self,zmin=1e-5,thetamin=1e-5,thetamax=0.4,mmin=1e-5,mmax=1e3,isize=1):
        z = beta.rvs(0.1,1, size=isize)
        theta = beta.rvs(0.1,1, size=isize)
        sinthetarest=self.sinthetaR2(z,theta)
        restmass=self.mass(z,theta)
        count=0
        while(theta < thetamin or theta > thetamax or
              restmass < mmin  or restmass > mmax  or 
              np.isnan(sinthetarest) or np.abs(sinthetarest) > 1 or
              z < zmin):
            z = beta.rvs(0.1,1, size=isize)
            theta = beta.rvs(0.1,1, size=isize)
            sinthetarest=self.sinthetaR2(z,theta)
            restmass=self.mass(z,theta)
            count+=1
            if count > 1000:
                #sample [0.00761592] - 1e-05 < [0.00122907] 0.0009174540921386668 < [1.48784275e-05] 1e-05 < [0.31999996] < 0.4
                print("sample",sinthetarest,"-",mmin,"<",restmass,"<",mmax,zmin,"<",z,thetamin,"<",theta,"<",thetamax)
        #print("z:",z,"theta:",theta,"thetarest",sinthetarest)
        return restmass,sinthetarest

    def p2(self,iM1,iM2,iMM):
        """       
        #Phil, fix to ensure momentum is back to back and mass is perserved
        #sqrt(p^2+m1^2)+sqrt(p^2+m2^2)=mother.mom.m=> solve for p^2
        """
        return (iM1**4+iM2**4+iMM**4-2*(iM1*iM2)**2-2*(iM1*iMM)**2-2*(iM2*iMM)**2)/(2*iMM)**2

    def rotateTheta(self,idau,itheta):
        v1     = [idau.p_x,idau.p_y,idau.p_z]
        axis=[0,1,0]
        v1rot=np.dot(self.rotation_matrix(axis,itheta), v1)
        dau1_mom = Momentum4(idau.e,v1rot[0],v1rot[1],v1rot[2])
        return dau1_mom

    def rotatePhi(self,idau,itheta):
        v1     = [idau.p_x,idau.p_y,idau.p_z]
        axis=[1,0,0]
        v1rot=np.dot(self.rotation_matrix(axis,itheta), v1)
        dau1_mom = Momentum4(idau.e,v1rot[0],v1rot[1],v1rot[2])
        return dau1_mom
 

    def massapprox(self,z,theta,p):
        return np.sqrt(z*(1-z))*p*theta

    def massp(self,z,theta,p):
        return self.mass(z,theta)*p

    def dau2mass(self,mother,z,theta):
        

        dau1_m = self.massapprox(z,theta,mother.mom.p)
        dau1_e = (mother.mom.p)*mother.z
        dau1_eta = self.theta_to_eta(-mother.randtheta+np.pi/2)
        if self.verbose:
            print(dau1_eta,dau1_m,dau1_e)
        d1=Momentum4.e_m_eta_phi(dau1_e[0], dau1_m, dau1_eta[0],0)
        mo=Momentum4.e_m_eta_phi(mother.mom.e,mother.mom.m, 0,0)
        
        if self.verbose:
            print(dau1_e,dau1_m,mo,d1,(mo-d1).m,mo.m,d1.m)
        return ((mo-d1).m)/mother.mom.p

    def randztheta(self,mother,zmin=1e-5,thetamin=1e-5,thetamax=0.4,mmin=1e-10,mmax=0.2,isize=1):
        z = beta.rvs(0.1,1, size=isize)
        theta = beta.rvs(0.1,1, size=isize)
        restmass=self.mass(z,theta)
        dau2mass=self.dau2mass(mother,z,theta)
        count=0
        massmax = np.minimum(0.2,mmax)
        while(theta < thetamin or theta > thetamax    or
              restmass < mmin  or restmass > massmax  or
              dau2mass < mmin  or dau2mass > massmax  or
              z < zmin or z > 0.5):
            z = beta.rvs(0.1,1, size=isize)
            theta = beta.rvs(0.1,1, size=isize)
            restmass=self.mass(z,theta)
            dau2mass=self.dau2mass(mother,z,theta)
            count+=1
            if count > 1000:
                print("theta 1 ",restmass,massmax,z,theta,dau2mass,mother.z,mother.randtheta)
        if self.verbose:
            print("dau2mass",restmass*mother.mom.p,dau2mass*mother.mom.p,mother.randtheta)
        #restmass=self.mass(z,theta)
        #massapprox=self.massapprox(z,theta,1)
        #print("mass check",restmass,massapprox,"---",z,theta,zmin,mmin)
        return z,theta

    #x = sqrt(4 m^2 - sqrt(4 m^4 + 8 m^2 p^2 - 3 m1^4 + 6 m1^2 m2^2 - 12 m1^2 p^2 z + 6 m1^2 p^2 - 3 m2^4 + 12 m2^2 p^2 z - 6 m2^2 p^2 - 12 p^4 z^2 + 12 p^4 z + p^4) - 3 m1^2 - 3 m2^2 - 6 p^2 z^2 + 6 p^2 z + p^2)/sqrt(6)
    def randz(self,mother,thetamin=1e-10,thetamax=0.4,isize=1):
       zmin=0.2/mother.mom.p,
       m=mother.mom.m/mother.mom.p
       mmax = np.minimum(mother.mom.m/mother.mom.p,0.4)
       z = beta.rvs(0.1,1, size=isize)
       theta=m/np.sqrt(z*(1-z))
       #dau2mass=self.dau2mass(mother,z,theta)
       count=0
       while(theta < thetamin or theta > thetamax or z < zmin or z > 0.5):
           z = beta.rvs(0.1,1, size=isize)
           theta=m/np.sqrt(z*(1-z))
           #dau2mass=self.dau2mass(mother,z,theta)
           count+=1
           if count > 1000:
            print("theta",m,z,theta,np.sqrt(z*(1-z)))
       return z,theta
    
    def mom_value(self,e,z1,t1,z2,t2,z):
        c=1+z*0.5*z1*t1**2+(1-z)*0.5*z2*t2**2
        return e/c


    def checkm1m2m(self,m,m1,m2):
        v1=m**2-m1**2-m2**2
        v2=m1*np.sqrt(1+m2**2)
        return v1 > v2

    def fullform(self,z,theta):
        v1=z*(1-z)
        v2=theta**2
        v3=np.sqrt(z**2+theta**2)*np.sqrt((1-z)**2+theta**2)
        return np.sqrt(2*(v1+v2+v3))

    
    def theta_func(self,z,m,m1,m2,p):
        val0=(1./(4.*p**2))*(m**4-2*(m**2)*(m1**2+m2**2)+(m1**2-m2**2)**2)
        val1=z*(m1**2-m2**2)
        val2=m1**2
        val3=z*(1-z)*m**2
        num=(val0+val1-val2+val3)
        den=(p**2+m**2)
        return np.arctan(np.sqrt(num/den))

    def ptheta(self,z,m,m1,m2,p):
        first=4*m**2+p**2-3*m1**2-3*m2**2+6*z*(1-z)*p**2
        second=4*(m**2-2*p**2)*m**2-3*(m1**2-m2**2)**2+(6*p**2-12*z*p**2)*(m1**2-m2**2+2*z*p**2)+(12*z**2+1)*p**4
        return np.arctan(np.sqrt((first-np.sqrt(second))/6.))

    def dau2(self,iMother,iM1,iTheta,iZ,iPhi):
        # print('In dau2: mother.mom.e', iMother.mom.e)

        dau1_m  = iM1
        dau1_px = iZ*iMother.mom.p
        dau1_pz = np.tan(iTheta)*iMother.mom.p
        dau1_e  = np.sqrt(dau1_px**2+dau1_pz**2+dau1_m**2)
        dau1_theta = iTheta*np.cos(iPhi)
        dau1_phi   = iTheta*np.sin(iPhi)
        dau1_eta   = self.theta_to_eta(-dau1_theta+np.pi/2)
        dau1_e = np.sqrt(dau1_px**2+dau1_pz**2+dau1_m**2)
        if hasattr(dau1_e,"__len__"):
            dau1_e = dau1_e[0]
        if hasattr(dau1_phi,"__len__"):
            dau1_eta = dau1_eta[0]
        if hasattr(dau1_phi,"__len__"):
            dau1_phi = dau1_phi[0]
        if hasattr(dau1_m,"__len__"):
            dau1_m = dau1_m[0]
        d1=Momentum4.e_m_eta_phi(dau1_e, dau1_m, dau1_eta,dau1_phi)
        mo=iMother.mom
        d1 = self.rotateTheta(d1,iMother.mom.theta-np.pi/2)
        d1 = self.rotatePhi  (d1,iMother.mom.phi)
        d2 = mo-d1
        return d1,d2

    def daun(self, mother, n):
        daughters = []
        cur_mother = mother
        # Flatten mother.mom.e if needed:
        mother_e = mother.mom.e
        if hasattr(mother_e, "__len__"):
            mother_e = mother_e[0]
        if self.verbose:
            print('In daun function')
            print('Starting mother momentum', mother.mom)
        remaining_mom = mother.mom
        for i in range(n - 1):
            # print('Creating daughter number', i)
            randomdraw_phi = np.random.uniform(0,2*np.pi)
            zrand,randomdraw_theta,rand_m1,rand_m2=self.randz(mother=cur_mother,iPhi=randomdraw_phi,isize=1)
            # print('zrand',zrand)


            rand_m1 = rand_m1[0] if hasattr(rand_m1, "__len__") else rand_m1
            randomdraw_theta = randomdraw_theta[0] if hasattr(randomdraw_theta, "__len__") else randomdraw_theta
            zrand = zrand[0] if hasattr(zrand, "__len__") else zrand
            randomdraw_phi = randomdraw_phi[0] if hasattr(randomdraw_phi, "__len__") else randomdraw_phi
            dau1, dau2 = self.dau2(cur_mother, rand_m1, randomdraw_theta, zrand, randomdraw_phi)
            
            
            if dau1.e < 0:
                print('Negative energy daughter 1:', dau1)
            if dau2.e < 0:
                print('Negative energy daughter 2:', dau2)
            # print('Daughter 1', dau1)
            # print('Daughter 2', dau2)
            
            
            daughters.append(
                particle(
                    mom=dau1,
                    randtheta=randomdraw_theta,
                    z=zrand,
                    m1=-1000,
                    m2=-1000,
                    prong_label=mother.prong_label,
                    part_label=-1,
                    part_parent_label=mother.part_label,
                    resonance_origin=mother.resonance_origin
                )
            )
            cur_mother.mom = dau2
            remaining_mom -= dau1
            if self.verbose:
                print('Remaining mom', remaining_mom)
            # print('Cur mother mom check', cur_mother.mom)

        # Final daughter for momentum conservation
        daughters.append(
            particle(
                mom=remaining_mom,
                randtheta=-1000,
                z=0,
                m1=-1000,
                m2=-1000,
                prong_label=mother.prong_label,
                part_label=-1,
                part_parent_label=mother.part_label,
                resonance_origin=mother.resonance_origin
            )
        )
        return daughters

    def checkdau2(self,iMother,iM1,iTheta,iZ,iPhi):
        d1,d2=self.dau2(iMother,iM1,iTheta,iZ,iPhi)
        # print('In checkdau2')
        # print(iMother, iM1, iTheta, iZ, iPhi)
        # print(d1, d2)
        # print('D2 complex mass check', np.iscomplex(d2.m)) 
        return np.iscomplex(d2.m)

    def randz(self,mother,iPhi,thetamin=1e-10,thetamax=0.4,isize=1):
        m=mother.mom.m
        p=mother.mom.p
        zmin=np.maximum(0.2/mother.mom.p,0.5*(1-np.sqrt(1-(m/p)**2)))
        zmax=0.5
        z  = beta.rvs(0.1,1, size=isize)
        z1 = beta.rvs(0.1,1, size=isize)
        t1 = beta.rvs(0.1,1, size=isize)
        z2 = beta.rvs(0.1,1, size=isize)
        t2 = beta.rvs(0.1,1, size=isize)
        m1 = self.fullform(z1,t1)*p*z
        m2 = self.fullform(z2,t2)*p*(1-z)
        theta=self.theta_func(z,m,m1,m2,p)
        count=0
        # print(mother, iPhi, thetamin, thetamax, isize)
        # print(theta,z,m1,m2,self.checkm1m2m(m,m1,m2),self.checkdau2(mother,m1,theta,z,iPhi))
        while(theta < thetamin or theta > thetamax or (np.isnan(theta)) or z < zmin or z > zmax
              or z1 > 0.5 or z > 0.5 or t1 > 0.5 or t2 > 0.5
              or m1 < 0.1 or m2 < 0.1 
              or (not self.checkm1m2m(m,m1,m2)) or self.checkdau2(mother,m1,theta,z,iPhi)
                ):
          z  = beta.rvs(0.1,1, size=isize)
          z1 = beta.rvs(0.1,1, size=isize)
          t1 = beta.rvs(0.1,1, size=isize)
          z2 = beta.rvs(0.1,1, size=isize)
          t2 = beta.rvs(0.1,1, size=isize)
          m1 = self.fullform(z1,t1)*p*z
          m2 = self.fullform(z2,t2)*p*(1-z)
          theta=self.theta_func(z,m,m1,m2,p)
          count+=1
          if count > 1000:
            #print("Sampled more than 1000 times")
            #print("theta",m,p,z,theta,m1,m2,self.checkm1m2m(m,m1,m2),self.checkdau2(mother,m1,theta,z,iPhi))
            return -1,-1,-1,-1
        return z,theta,m1,m2

    def create_resonance(self, mother, resonance):
        resonance_mass = resonance['mass']
        decay_products = resonance['decay_products']

        # -- Flatten mother 4-vector if it is a length-1 array --
        mother_e = mother.mom.e
        mother_m = mother.mom.m
        mother_eta = mother.mom.eta
        mother_phi = mother.mom.phi
        mother_p   = mother.mom.p

        # If any are length-1 arrays, convert them to scalars
        # (the 'hasattr(x, "__len__")' check is just to see if it’s array-like)
        if hasattr(mother_e, "__len__"):
            mother_e = mother_e[0]
        if hasattr(mother_eta, "__len__"):
            mother_eta = mother_eta[0]
        if hasattr(mother_phi, "__len__"):
            mother_phi = mother_phi[0]
        if hasattr(mother_p, "__len__"):
            mother_p = mother_p[0]

        # Ensure mother has enough energy for decay
        

        # Decay from quark -> resonance + quark
        # randomdraw_phi = np.random.uniform(0,2*np.pi)
        # zrand,randomdraw_theta,quark_mass=self.randz_resonance(mother=mother,iPhi=randomdraw_phi,fixed_m1 = resonance_mass)
        # print('Quark mass', quark_mass)
       

        # resonance_mom, quark_mom = self.dau2(mother,resonance_mass,randomdraw_theta,zrand,randomdraw_phi)
        if self.verbose:
            print('Hard splitting mother:',mother)
            print('Mother momentum', mother.mom)
            print('Mother mass', mother.mom.m)
            print('Resonance mass:',resonance_mass)
        resonance_part, quark_part,_,_ = self.hardsplit_to_resonance(mother,resonance_mass)
        count = 0
        resonance_mom = resonance_part.mom
        cur_resonance_mass = resonance_mom.m
        cur_quark_mass = quark_part.mom.m
        while cur_resonance_mass < 0.1 or cur_quark_mass < 0.1 or np.isnan(cur_resonance_mass) or np.isnan(cur_quark_mass) or np.iscomplex(cur_resonance_mass) or np.iscomplex(cur_quark_mass):
            resonance_part, quark_part,_,_ = self.hardsplit_to_resonance(mother,resonance_mass)
            resonance_mom = resonance_part.mom
            cur_resonance_mass = resonance_mom.m
            cur_quark_mass = quark_part.mom.m
            count += 1
            if count > 10000:
                print('Failed to find splitting')
                return -1, -1, -1
        if self.verbose:
            print('Resonance mass', resonance_mom.m)
            print('Resonance momentum', resonance_mom)
            print('Quark mass', quark_part.mom.m)
            print('Quark momentum', quark_part.mom)
        

        resonance_particle = particle(
            mom=resonance_mom,
            randtheta=-1000,
            z=-1000,
            m1=-1000,
            m2=-1000,
            prong_label=mother.prong_label,
            part_label=self.total_part_counter,
            part_parent_label=mother.part_label,
            resonance_origin=f"Resonance_{resonance_mass}_{self.n_resonance}"
        )

        self.total_part_counter += 1
        quark_part.part_label = self.total_part_counter
        self.total_part_counter += 1
        quark_part.part_parent_label = mother.part_label
        quark_part.prong_label = mother.prong_label
        if self.verbose:
            print('Resonance created:', resonance_particle)
            print('Massless Quark created:', quark_part)

            print('Showering resonance \n')
        # Decay the resonance using daun
        daughters = self.daun(copy.deepcopy(resonance_particle), decay_products)
        
        self.n_resonance += 1
        # print('Checking for -1')
        # print('Resonance:',resonance_particle)
        # print('Quark:',quark_part)
        return daughters, quark_part, resonance_particle


   
    def softsplit(self, mother):
        """Soft splitting logic with resonance integration."""
        
        # print('Softsplitting with', mother.mom.eta, mother.mom.phi, mother.mom.p)

        randomdraw_phi = np.random.uniform(0,2*np.pi)
        randomdraw_theta=mother.randtheta
        zrand=mother.z
        rand_m1=mother.m1
        rand_m2=mother.m2
        if randomdraw_theta == -1000:
            zrand,randomdraw_theta,rand_m1,rand_m2=self.randz(mother=mother,iPhi=randomdraw_phi,isize=1)
            mother.z=zrand
            mother.randtheta=randomdraw_theta
            if zrand == -1:
                dau1 = particle(mom=mother.mom,randtheta=-1000,z=-1,m1=-1000,m2=-1000, 
                                prong_label=-1,part_label=-1,part_parent_label=-1, resonance_origin = 'None')
                dau2 = particle(mom=mother.mom,randtheta=-1000,z=-1,m1=-1000,m2=-1000, 
                                prong_label=-1,part_label=-1,part_parent_label=-1, resonance_origin = 'None')
                
                return [dau1,dau2], -111.11, -111.11
        dau1_mom,dau2_mom=self.dau2(mother,rand_m1,randomdraw_theta,zrand,randomdraw_phi)
        dau1 = particle(mom=dau1_mom,randtheta=-1000,z=-1000,m1=-1000,m2=-1000, 
                        prong_label=-1,part_label=-1,part_parent_label=mother.part_label, resonance_origin = 'None')
        dau2 = particle(mom=dau2_mom,randtheta=-1000,z=-1000,m1=-1000,m2=-1000, 
                        prong_label=-1,part_label=-1,part_parent_label=mother.part_label, resonance_origin = 'None')


        dau1_pt = dau1.mom.p_t
        dau2_pt = dau2.mom.p_t
        if hasattr(dau1_pt, "__len__"):
            dau1_pt = dau1_pt[0]
        if hasattr(dau2_pt, "__len__"):
            dau2_pt = dau2_pt[0]
        if hasattr(randomdraw_theta, "__len__"):
            randomdraw_theta = randomdraw_theta[0]

        self.z.append(np.min([dau1_pt, dau2_pt])/(dau1_pt + dau2_pt))
        self.zsoft.append(np.min([dau1_pt, dau2_pt])/(dau1_pt + dau2_pt))
        self.randtheta.append(randomdraw_theta)
        #print(dau1, dau2, np.min([dau1_pt, dau2_pt])/(dau1_pt + dau2_pt), randomdraw_theta)
        #print("randomtheta: ", randomdraw_theta[0])
        return [dau1, dau2], np.min([dau1_pt, dau2_pt])/(dau1_pt + dau2_pt), randomdraw_theta

    
    def hardsplit(self, mother, nthsplit):
        #Hard splitting performed in the rest frame of the mother, rotated, and then lorentz boosted back
        #Hard splitting prior: Gaussian around pi/2,
        np.random.seed()
        #randomdraw_theta = np.abs(np.random.normal(np.pi/2,0.1))
        randomdraw_theta = np.random.uniform(0.1,np.pi/2.-0.1)
        randomdraw_phi   = np.random.uniform(0,2*np.pi)
        if nthsplit==1:
            randomdraw_phi = 0
        #print("hard", nthsplit," ", mother.m)
        dau1_m = np.random.uniform(mother.mom.m/16, mother.mom.m/2)
        dau2_m = np.random.uniform(mother.mom.m/16, mother.mom.m/2)
        #if nthsplit == 1 and self.nprong > 2:
        #    dau1_m = 80.379
        #    dau2_m = 40.18
        #else:  
        #    dau1_m = 40.18
        #    dau2_m = 40.18
        if nthsplit == 1:
            dau1_m = 80.379
            dau2_m = 40.18
            
        if nthsplit == 2:
            dau1_m = 40.18
            dau2_m = 40.18
            
        dau1_theta = (np.pi/2 + randomdraw_theta)
        dau2_theta = (np.pi/2 - randomdraw_theta)        
        dau1_phi = mother.mom.phi + randomdraw_phi
        dau2_phi = mother.mom.phi + randomdraw_phi + np.pi
        dau1_phi %= (2*np.pi)
        dau2_phi %= (2*np.pi)
        #prep for 4-vector
        dau_p2   = self.p2(dau1_m,dau2_m,mother.mom.m)
        dau1_e   = np.sqrt(dau_p2+dau1_m**2)
        dau2_e   = np.sqrt(dau_p2+dau2_m**2)
        dau1_mom = Momentum4.e_m_eta_phi(dau1_e, dau1_m, self.theta_to_eta(dau1_theta), dau1_phi)
        dau2_mom = Momentum4.e_m_eta_phi(dau2_e, dau2_m, self.theta_to_eta(dau2_theta), dau2_phi)
        dau1_mom = self.rotateTheta(dau1_mom,mother.mom.theta-np.pi/2)
        dau2_mom = self.rotateTheta(dau2_mom,mother.mom.theta-np.pi/2)
        dau1_mom = self.rotatePhi(dau1_mom,mother.mom.phi)
        dau2_mom = self.rotatePhi(dau2_mom,mother.mom.phi)
        dau1 = particle(mom=dau1_mom,randtheta=-1000,z=-1000,m1=-1000,m2=-1000, 
                        prong_label=-1,part_label=-1,part_parent_label=-1, resonance_origin = 'None')
        dau2 = particle(mom=dau2_mom,randtheta=-1000,z=-1000,m1=-1000,m2=-1000, 
                        prong_label=-1,part_label=-1,part_parent_label=-1, resonance_origin = 'None')
        dau1.mom = dau1.mom.boost_particle(mother.mom)
        dau2.mom = dau2.mom.boost_particle(mother.mom)
        self.randtheta.append(randomdraw_theta)
        self.zhard.append(np.min([dau1.mom.p_t, dau2.mom.p_t])/(dau1.mom.p_t+dau2.mom.p_t))
        self.z.append(np.min([dau1.mom.p_t, dau2.mom.p_t])/(dau1.mom.p_t+dau2.mom.p_t))
        return dau1, dau2, np.min([dau1.mom.p_t, dau2.mom.p_t])/(dau1.mom.p_t+dau2.mom.p_t), randomdraw_theta

    def hardsplit_to_resonance(self, mother, resonance_mass):
        #Hard splitting performed in the rest frame of the mother, rotated, and then lorentz boosted back
        #Hard splitting prior: Gaussian around pi/2,
        np.random.seed()
        #randomdraw_theta = np.abs(np.random.normal(np.pi/2,0.1))
        randomdraw_theta = np.random.uniform(0.1,np.pi/2.-0.1)
        randomdraw_phi   = np.random.uniform(0,2*np.pi)
        
        dau1_m = resonance_mass
        dau2_m = np.random.uniform(mother.mom.m/16, mother.mom.m/2)

        dau1_theta = (np.pi/2 + randomdraw_theta)
        dau2_theta = (np.pi/2 - randomdraw_theta)        
        dau1_phi = mother.mom.phi + randomdraw_phi
        dau2_phi = mother.mom.phi + randomdraw_phi + np.pi
        dau1_phi %= (2*np.pi)
        dau2_phi %= (2*np.pi)
        #prep for 4-vector
        dau_p2   = self.p2(dau1_m,dau2_m,mother.mom.m)
        dau1_e   = np.sqrt(dau_p2+dau1_m**2)
        dau2_e   = np.sqrt(dau_p2+dau2_m**2)
        # Ensure inputs are floats
        if hasattr(dau1_e, "__len__"):
            dau1_e = dau1_e[0]
        if hasattr(dau1_m, "__len__"):
            dau1_m = dau1_m[0]
        if hasattr(dau1_theta, "__len__"):
            dau1_theta = dau1_theta[0]
        if hasattr(dau1_phi, "__len__"):
            dau1_phi = dau1_phi[0]
        if hasattr(dau2_e, "__len__"):
            dau2_e = dau2_e[0]
        if hasattr(dau2_m, "__len__"):
            dau2_m = dau2_m[0]
        if hasattr(dau2_theta, "__len__"):
            dau2_theta = dau2_theta[0]
        if hasattr(dau2_phi, "__len__"):
            dau2_phi = dau2_phi[0]
        dau1_eta = self.theta_to_eta(dau1_theta)
        dau2_eta = self.theta_to_eta(dau2_theta)

        if hasattr(dau1_eta, "__len__"):
            dau1_eta = dau1_eta[0]
        if hasattr(dau2_eta, "__len__"):
            dau2_eta = dau2_eta[0]

        dau1_mom = Momentum4.e_m_eta_phi(dau1_e, dau1_m, dau1_eta, dau1_phi)
        dau2_mom = Momentum4.e_m_eta_phi(dau2_e, dau2_m, dau2_eta, dau2_phi)
        dau1_mom = self.rotateTheta(dau1_mom,mother.mom.theta-np.pi/2)
        dau2_mom = self.rotateTheta(dau2_mom,mother.mom.theta-np.pi/2)
        dau1_mom = self.rotatePhi(dau1_mom,mother.mom.phi)
        dau2_mom = self.rotatePhi(dau2_mom,mother.mom.phi)
        dau1 = particle(mom=dau1_mom,randtheta=-1000,z=-1000,m1=-1000,m2=-1000, 
                        prong_label=-1,part_label=-1,part_parent_label=-1, resonance_origin = 'None')
        dau2 = particle(mom=dau2_mom,randtheta=-1000,z=-1000,m1=-1000,m2=-1000, 
                        prong_label=-1,part_label=-1,part_parent_label=-1, resonance_origin = 'None')
        dau1.mom = dau1.mom.boost_particle(mother.mom)
        dau2.mom = dau2.mom.boost_particle(mother.mom)
        self.randtheta.append(randomdraw_theta)
        self.zhard.append(np.min([dau1.mom.p_t, dau2.mom.p_t])/(dau1.mom.p_t+dau2.mom.p_t))
        self.z.append(np.min([dau1.mom.p_t, dau2.mom.p_t])/(dau1.mom.p_t+dau2.mom.p_t))
        return dau1, dau2, np.min([dau1.mom.p_t, dau2.mom.p_t])/(dau1.mom.p_t+dau2.mom.p_t), randomdraw_theta


    def draw_first_particle(self):
        #Draw mass from pdf
        np.random.seed()
        if self.massprior == "signal":
            m = np.random.normal(172.76, 1.32)
            if self.nprong == 4:
                m = np.random.normal(500, 1.32)
                

        if self.massprior == "background":
            m = np.random.uniform(0, 100)
        
        p = np.random.exponential(400)
        #delete later
        if self.doFixP:
            p = 400
        vec0 = Momentum4.m_eta_phi_p(m, 0, 0, p)
        part = particle(mom=vec0,randtheta=-1000,z=-1000,m1=-1000,m2=-1000, prong_label=-1,part_label=-1,part_parent_label=-1, resonance_origin = 'None')  
        return part

    def hard_decays(self):
        hardparticle_list = [self.draw_first_particle()]
        prong = 1
        zlist = []
        thetalist = []
        while prong < self.nprong:
            dau1, dau2, z, theta = self.hardsplit(hardparticle_list[0],prong)
            
            #print(dau1.mom.m, dau2.mom.m)
            hardparticle_list.pop(0)
            self.reverse_insort(hardparticle_list, dau1)
            self.reverse_insort(hardparticle_list, dau2)
            zlist.append(z)
            thetalist.append(theta)
            prong += 1
            
        return hardparticle_list, zlist, thetalist

    def genshower(self, _):
        """Generate a full shower including soft splittings and resonances."""
        # Hard decay until one particle for each prong
        self.total_part_counter = 0
        self.n_resonance = 0
        showered_list, zlist, thetalist = self.hard_decays()

        all_particles_list = []

        for i in range(len(showered_list)):
            showered_list[i].prong_label = i
            showered_list[i].part_label = i
            showered_list[i].part_parent_label = -1
            # all_particles_list.append(showered_list[i])
        
        self.total_particle = len(showered_list)
        self.total_part_counter = len(showered_list)
        if self.verbose:
            print('Before soft splitting')
            print(all_particles_list)

        while self.total_particle < self.nparticle:
            if self.verbose:
                print('\nNew Splitting')
            if showered_list[0].mom.p < 1:
                break
            if self.verbose:
                print('Splitting particle', showered_list[0])

            np.random.seed()
            resonance_made = False
            if self.n_resonance < self.max_resonance_per_jet:
                for resonance in self.resonance_data:
                    if np.random.rand() < resonance['probability']:
                        if showered_list[0].mom.m > resonance['mass']*1.2:
                            resonance_made = True
                            daughters, intermediate_quark, resonance_particle =  self.create_resonance(showered_list[0], resonance)
                            z, theta = -1000, -1000

            if not resonance_made:
                daughters, z, theta = self.softsplit(showered_list[0])
            
            # print(f'Showered into {len(daughters)-1} daughters')
            if daughters[0].z == -1:
                break
            
            for dau in daughters:
                
                dau.part_label = self.total_part_counter
                self.total_part_counter += 1
                dau.prong_label = showered_list[0].prong_label
                if resonance_made:
                    dau.part_parent_label = resonance_particle.part_label
                else:
                    dau.part_parent_label = showered_list[0].part_label
                if self.verbose:
                    print('Daughter:', dau)
                    print('Daughter mass', dau.mom.m)
                if dau.resonance_origin == 'None':
                    dau.resonance_origin = showered_list[0].resonance_origin

            all_particles_list.append(showered_list[0])
            showered_list.pop(0)
            if resonance_made:
                if self.verbose:
                    print('intermediate_quark part label', intermediate_quark.part_label)
                    print('resonance_particle part label', resonance_particle.part_label)
                    
                    print('intermediate_quark', intermediate_quark)
                    print('resonance_particle', resonance_particle)
                all_particles_list.append(resonance_particle)
                self.reverse_insort(showered_list, intermediate_quark)

            for dau in daughters:
                # print('Inserting daughter into showered_list', dau)
                self.reverse_insort(showered_list, dau)
            
            # print('len of showered_list', len(showered_list))
            # print('self.total_particle', total_particle)
            # print('self.nparticle', self.nparticle)

            zlist.append(z)
            thetalist.append(theta)
            if resonance_made:
                self.total_particle += len(daughters) # len(daughters) for resonance -> daughters, +1 for intermediate quark, -1 for showered particle
            else:
                self.total_particle += len(daughters) - 1 # len(daughters) for resonance -> daughters, -1 for showered particle
            
            
            # print(total_particle)

        for p in showered_list:
            all_particles_list.append(p)

        return self.total_particle, showered_list, zlist, thetalist, all_particles_list

    def shower(self,_):
        i=0

        total_particle,showered_list,zlist, thetalist, all_particles_list=self.genshower(i)
        #print(total_particle, self.nparticle)
        while total_particle <  self.nparticle:
            total_particle,showered_list,zlist, thetalist, all_particles_list=self.genshower(i)
        # arr = []

        # check = Momentum4(0,0,0,0)
        # for j in range(self.nparticle):
        #     # Debug prints to inspect the state of showered_list and its elements
        #     # print(f"showered_list[{j}]: {showered_list[j]}")
        #     # print(f"showered_list[{j}].mom: {showered_list[j].mom}")
        #     # print(f"showered_list[{j}].mom.p_t: {showered_list[j].mom.p_t}")
        #     # print(f"showered_list[{j}].mom.eta: {showered_list[j].mom.eta}")
        #     # print(f"showered_list[{j}].mom.phi: {showered_list[j].mom.phi}")

        #     arr.append(showered_list[j].mom.p_t)
        #     arr.append(showered_list[j].mom.eta)
        #     arr.append(showered_list[j].mom.phi)
        #     check += showered_list[j].mom


        #print("squeeze",np.squeeze(np.array(arr)).shape, np.squeeze(np.array(zlist)).shape, np.squeeze(np.array(thetalist)).shape)
        return np.array(showered_list),np.array(all_particles_list)
    
    def generate_dataset(self, nevent,verbose = False):
        self.verbose = verbose

        #output = torch.FloatTensor([])
        
#         data = np.empty([nevent, 3*self.nparticle], dtype=float)
#         data_z     = np.empty([nevent, self.nparticle-1], dtype=float)
#         data_theta = np.empty([nevent, self.nparticle-1], dtype=float)
#         data_particles = np.empty([nevent, self.nparticle], dtype=object)
                
        data = np.empty([nevent], dtype=object)
        data_particles = np.empty([nevent], dtype=object)
        data_shower_particles = np.empty([nevent], dtype=object)
        base_n_particle = self.nparticle
#         data = []
#         data_z = []
#         data_theta = []
#         data_particles = []
        
        i =0 
        if self.doMultiprocess:
            pool = Pool(processes=self.ncore)
            data, data_z, data_theta  = zip(*pool.map(self.shower,range(nevent)))

        else:
            while i < nevent:
                adj = 2*int(np.random.normal(0, int(0.1*base_n_particle)))
                self.nparticle = base_n_particle + adj
                if i % 10 == 0:
                    print("event :",i)
                try:
                    arr_particles, shower_particles = self.shower(i)    
                    data_particles[i] = arr_particles
                    data_shower_particles[i] = shower_particles
                    i += 1
                except:
                    print("Error in event :",i)
                    
         
        
        #return output
        return data_particles, data_shower_particles
        
        
#         return np.array(data), np.array(data_z), np.array(data_theta), np.array(data_particles)

    def mother_2_to_1(self,d1,d2):
        # mother 4 vector: mo = d1 + d2
        m0 = d1+d2
        return m0

    def softcombine(self,dau1,dau2):
        mother_mom = self.mother_2_to_1(dau1.mom, dau2.mom)
        mother = particle(mom=mother_mom,randtheta=-1000,z=-1000,m1=-1000,m2=-1000)
        return mother
    #def visualize_one_event(self):

