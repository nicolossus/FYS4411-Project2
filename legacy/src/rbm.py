# 2-electron VMC code for 2dim quantum dot with importance sampling
# Using gaussian rng for new positions and Metropolis- Hastings
# Added restricted boltzmann machine method for dealing with the wavefunction
# RBM code based heavily off of:
# https://github.com/CompPhysics/ComputationalPhysics2/tree/gh-pages/doc/Programs/BoltzmannMachines/MLcpp/src/CppCode/ob
from math import exp, sqrt
from random import random, seed, normalvariate
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import sys, time

import pandas as pd
from pandas import DataFrame

np.random.seed(4411)
# Changed from math.exp, sqrt etc to np.exp, sqrt ...

# Trial wave function for the 2-electron quantum dot in two dims
def WaveFunction(NumberParticles, Dimension, NumberHidden, r, a, b, w):

    #sigma=1.0           # From Morten
    #sig2 = sigma**2     # From Morten

    Psi1 = 0.0
    Psi2 = 1.0
    Q = Qfac(NumberHidden, r, b, w)

    for iq in range(NumberParticles):
        for ix in range(Dimension):
            Psi1 += (r[iq,ix]-a[iq,ix])**2 #Gaussian

    for ih in range(NumberHidden):
        Psi2 *= (1.0 + np.exp(Q[ih]))
    #print(Psi2)

    #Psi1 = np.exp(-Psi1/(2*sig2)) # From Morten
    Psi1 = np.exp(-Psi1 / 2)

    return Psi1*Psi2

def LocalEnergy(NumberParticles, Dimension, NumberHidden, r, a, b, w, interaction):

    #sigma=1.0           # From Morten
    #sig2 = sigma**2     # From Morten

    locenergy = 0.0
    kinetic_energy = 0.0
    potential_energy = 0.0

    Q = Qfac(NumberHidden, r, b, w)

    for iq in range(NumberParticles):
        for ix in range(Dimension):
            sum1 = 0.0
            sum2 = 0.0
            for ih in range(NumberHidden):
                sum1 += w[iq,ix,ih]/(1+np.exp(-Q[ih]))
                sum2 += w[iq,ix,ih]**2 * np.exp(Q[ih]) / (1.0 + np.exp(Q[ih]))**2
                #sum2 += w[iq,ix,ih]**2 * np.exp(-Q[ih]) / (1.0 + np.exp(-Q[ih]))**2 # ???

            #dlnpsi1 = -(r[iq,ix] - a[iq,ix]) /sig2 + sum1/sig2 # From Moren
            #dlnpsi2 = -1/sig2 + sum2/sig2**2                   # From Morten

            dlnpsi1 = -(r[iq, ix] - a[iq, ix]) + sum1
            dlnpsi2 = -1 + sum2
            #locenergy += 0.5*(-dlnpsi1*dlnpsi1 - dlnpsi2 + r[iq,ix]**2)
            kinetic_energy += -0.5*(dlnpsi1*dlnpsi1 + dlnpsi2)
            potential_energy += 0.5*r[iq,ix]**2

    if(interaction==True):
        for iq1 in range(NumberParticles):
            for iq2 in range(iq1):
                distance = 0.0
                for ix in range(Dimension):
                    distance += (r[iq1,ix] - r[iq2,ix])**2

                locenergy += 1 / np.sqrt(distance)
    #print("grad:", dlnpsi1)
    locenergy = kinetic_energy + potential_energy
    return locenergy

# Derivate of wave function ansatz as function of variational parameters
def DerivativeWFansatz(NumberHidden, r, a, b, w):

    #sigma=1.0           # From Morten
    #sig2 = sigma**2     # From Morten

    Q = Qfac(NumberHidden, r, b, w)

    denominator = 1 + np.exp(-Q)   # To reduce FLOPS

    # More const efficient than Morten
    WfDer_a = r - a
    WfDer_b = 1 / denominator
    WfDer_w = w / denominator

    return  WfDer_a, WfDer_b, WfDer_w

# Setting up the quantum force for the two-electron quantum dot, recall that it is a vector
def QuantumForce(NumberParticles, Dimension, NumberHidden, r, a, b, w):
    # r will be PositionOld/PositionNew

    #sigma=1.0           # From Morten
    #sig2 = sigma**2     # From Morten

    qforce = np.zeros((NumberParticles, Dimension), np.double)
    sum1 = np.zeros((NumberParticles, Dimension), np.double)

    Q = Qfac(NumberHidden, r, b, w)

    for ih in range(NumberHidden):
        sum1 += w[:,:,ih]/(1+np.exp(-Q[ih]))

    #qforce = 2*(-(r-a)/sig2 + sum1/sig2) # From Morten
    qforce = 2 * (-(r - a) + sum1)

    return qforce

def Qfac(NumberHidden, r, b, w):
    # h will be set to NumberHidden

    Q = np.zeros((NumberHidden), np.double)
    temp = np.zeros((NumberHidden), np.double)

    for ih in range(NumberHidden):
        temp[ih] = (r*w[:,:,ih]).sum()

    Q = b + temp

    return Q

def EnergyMinimization(NumberParticles, Dimension, NumberHidden, NumberMCcycles, a, b, w, outputfile, ef, interaction=True):
    # TODO Change names & structure
    # ef
    # interacton

    #NumberMCcycles= 10000  # From morten
    # Parameters in the Fokker-Planck simulation of the quantum force
    D = 0.5
    TimeStep = 0.05
    # positions
    PositionOld = np.zeros((NumberParticles, Dimension), np.double)
    PositionNew = np.zeros((NumberParticles, Dimension), np.double)
    # Quantum force
    QuantumForceOld = np.zeros((NumberParticles, Dimension), np.double)
    QuantumForceNew = np.zeros((NumberParticles, Dimension), np.double)

    # seed for rng generator
    #seed() # From Morten
    energy = 0.0
    DeltaE = 0.0

    # TODO Make this more compact / effective
    EnergyDer = np.empty((3,),dtype=object)
    DeltaPsi = np.empty((3,),dtype=object)
    DerivativePsiE = np.empty((3,),dtype=object)
    EnergyDer = [np.copy(a),np.copy(b),np.copy(w)]
    DeltaPsi = [np.copy(a),np.copy(b),np.copy(w)]
    DerivativePsiE = [np.copy(a),np.copy(b),np.copy(w)]
    for i in range(3): EnergyDer[i].fill(0.0)
    for i in range(3): DeltaPsi[i].fill(0.0)
    for i in range(3): DerivativePsiE[i].fill(0.0)

    #print(EnergyDer)


    #Initial position
    for i in range(NumberParticles):
        for j in range(Dimension):
            PositionOld[i, j] = normalvariate(0.0, 1.0) * np.sqrt(TimeStep)
    #print(PositionOld)
    wfold = WaveFunction(NumberParticles, Dimension, NumberHidden, PositionOld, a, b, w)
    QuantumForceOld = QuantumForce(NumberParticles, Dimension, NumberHidden, PositionOld, a, b, w)

    #Loop over MC MCcycles
    for cycle in range(NumberMCcycles):
        #Trial position moving one particle at the time
        for i in range(NumberParticles):
            for j in range(Dimension):
                PositionNew[i,j] = PositionOld[i,j]+normalvariate(0.0,1.0)*np.sqrt(TimeStep)+\
                                       QuantumForceOld[i,j]*TimeStep*D

            wfnew = WaveFunction(NumberParticles, Dimension, NumberHidden, PositionNew, a, b, w)
            QuantumForceNew = QuantumForce(NumberParticles, Dimension, NumberHidden, PositionNew, a, b, w)

            GreensFunction = 0.0
            for j in range(Dimension):
                GreensFunction += 0.5*(QuantumForceOld[i,j]+QuantumForceNew[i,j])*\
                                      (D*TimeStep*0.5*(QuantumForceOld[i,j]-QuantumForceNew[i,j])-\
                                      PositionNew[i,j]+PositionOld[i,j])

            GreensFunction = np.exp(GreensFunction)
            ProbabilityRatio = GreensFunction*wfnew**2/wfold**2
            #Metropolis-Hastings test to see whether we accept the move
            np.random.seed(5)
            #if random() <= ProbabilityRatio: # From Morten where random from random lib
            if np.random.uniform(0.0, 1.0) <= ProbabilityRatio:
                for j in range(Dimension):
                    PositionOld[i,j] = PositionNew[i,j]
                    QuantumForceOld[i,j] = QuantumForceNew[i,j]
                wfold = wfnew
        #print("wf new:        ", wfnew)
        #print("force on 1 new:", QuantumForceNew[0,:])
        #print("pos of 1 new:  ", PositionNew[0,:])
        #print("force on 2 new:", QuantumForceNew[1,:])
        #print("pos of 2 new:  ", PositionNew[1,:])
        DeltaE = LocalEnergy(NumberParticles, Dimension, NumberHidden, PositionOld, a, b, w, interaction)
        DerPsi = DerivativeWFansatz(NumberHidden, PositionOld, a, b, w)

        # TODO Make this more efficient XXX
        if cycle > NumberMCcycles * ef:
            DeltaPsi[0] += DerPsi[0]
            DeltaPsi[1] += DerPsi[1]
            DeltaPsi[2] += DerPsi[2]

            energy += DeltaE

            # TODO CHANGE
            outputfile.write(f"{(energy / (cycle - int(NumberMCcycles * ef) + 1.0)):f}")
            outputfile.write("\n")

            #print(f"{(energy / (cycle - int(NumberMCcycles * ef) + 1.0)):f}")

            # TODO Make this more efficient
            DerivativePsiE[0] += DerPsi[0]*DeltaE
            DerivativePsiE[1] += DerPsi[1]*DeltaE
            DerivativePsiE[2] += DerPsi[2]*DeltaE

    # Mean value calculation
    fraq = NumberMCcycles - (NumberMCcycles * ef)
    energy /= fraq # Morten divides by NumberMCcycles

    DerivativePsiE[0] /= fraq # Morten divides by NumberMCcycles
    DerivativePsiE[1] /= fraq # Morten divides by NumberMCcycles
    DerivativePsiE[2] /= fraq # Morten divides by NumberMCcycles

    DeltaPsi[0] /= fraq # Morten divides by NumberMCcycles
    DeltaPsi[1] /= fraq # Morten divides by NumberMCcycles
    DeltaPsi[2] /= fraq # Morten divides by NumberMCcycles

    EnergyDer[0]  = 2*(DerivativePsiE[0]-DeltaPsi[0]*energy)
    #print("\nEnergy Der0", EnergyDer[0])
    EnergyDer[1]  = 2*(DerivativePsiE[1]-DeltaPsi[1]*energy)
    #print("Energy Der1", EnergyDer[1])
    EnergyDer[2]  = 2*(DerivativePsiE[2]-DeltaPsi[2]*energy)
    #print("Energy Der2", EnergyDer[2])

    return energy, EnergyDer


def nodes_and_weights(NumberParticles, Dimension, NumberHidden):

    a = np.random.normal(loc = .0, scale = .5, size = (NumberParticles, Dimension))                # length x
    b = np.random.normal(loc = .0, scale = .5, size = (NumberHidden))                              # length h
    w = np.random.normal(loc = .0, scale = .5, size = (NumberParticles, Dimension, NumberHidden))  # M x N

    return a, b, w

def run_simulation(eta, MaxIterations, NumberMCcycles, interaction=True):

    # TODO Change names & Structure
    # eta is learning Rate
    # MaxIterations is max iterations
    #
    # filename
    # interaction


    NumberParticles = 1
    Dimension = 1
    NumberHidden = 2

    # TODO
    ef = 0.1
    gamma = 0.9

    # guess for parameters
    a, b, w = nodes_and_weights(NumberParticles, Dimension, NumberHidden)

    # TODO Change
    #savefile is based upon whether we use interaction or not
    outputfile = open("interactions_" + str(eta) + "_" + str(MaxIterations) + ".txt", "w")

    # Set up iteration using stochastic gradient method
    Energy = 0
    EDerivative = np.empty((3,),dtype=object)
    EDerivative = [np.copy(a),np.copy(b),np.copy(w)]

    # From Morten
    # Learning rate eta, max iterations, need to change to adaptive learning rate
    #eta = 0.001
    #MaxIterations = 50
    #iter = 0

    np.seterr(invalid='raise')
    Energies = np.zeros(MaxIterations)
    #EnergyDerivatives1 = np.zeros(MaxIterations) # From Morten
    #EnergyDerivatives2 = np.zeros(MaxIterations) # From Morten

    # TODO Change & Structure
    time_array = np.zeros(MaxIterations)

    mmt_a = np.zeros_like(a) # Momentums # TODO Change to one list
    mmt_b = np.zeros_like(b) # zeros_like only
    mmt_w = np.zeros_like(w)

    percentage = -1
    totaltime  = time.time()

    # TODO Change this ADD TQDM
    # Many changes from Morten
    for iter in range(MaxIterations):
        if int(100 * iter / MaxIterations) > percentage:
            percentage = int(100 * iter / MaxIterations)
            print(f"Completed {percentage}% ", end="")

        enditertime = time.time()

        Energy, EDerivative = EnergyMinimization(NumberParticles, Dimension, NumberHidden, NumberMCcycles, a-mmt_a*gamma,b-mmt_b*gamma,w-mmt_w*gamma, outputfile, ef, interaction=interaction)

        mmt_a = mmt_a * gamma + eta * EDerivative[0]
        mmt_b = mmt_b * gamma + eta * EDerivative[1]
        mmt_w = mmt_w * gamma + eta * EDerivative[2]

        a -= mmt_a; b -= mmt_b; w -= mmt_w

        Energies[iter] = Energy
        print(f"Energy: ", Energy)
        #EnergyDerivatives1[iter] = EDerivative[0] # From Morten
        #EnergyDerivatives2[iter] = EDerivative[1] # From Morten
        #EnergyDerivatives3[iter] = EDerivative[2] # From Morten
        time_array[iter] = time.time() - enditertime

    print(f"Completed 100% ")
    #nice printout with Pandas
    pd.set_option('max_columns', 6)
    data = {'Energy': Energies, 'Time': time_array} # From Morten
                                                    # ,'A Derivative':EnergyDerivatives1,'B Derivative':EnergyDerivatives2,'Weights Derivative':EnergyDerivatives3}

    print("Results of the interaction:") # If any ...

    frame = pd.DataFrame(data)
    print(frame)

    # TODO
    print(f"The mean Energy: {np.mean(Energies)}")
    print(f"Lowest Energy: {np.min(Energies)} | Highest Energy {np.max(Energies)}")
    print(F"Total time passed: {time.time() - totaltime} | Mean time per iter {np.mean(time_array)}")
    np.savetxt("energies_" + str(eta) + "_" + str(MaxIterations) + ".txt", Energies)
    outputfile.close()

run_simulation(0.05, 100, 10000, interaction=False)
