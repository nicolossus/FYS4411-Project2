import numpy as np
from numpy.random import default_rng
#from src import vmc, NonInteractRB
import matplotlib.pyplot as plt



def Qfac(NumberHidden, r, b, w):
    # h will be set to NumberHidden

    Q = np.zeros((NumberHidden), np.double)
    temp = np.zeros((NumberHidden), np.double)

    for ih in range(NumberHidden):
        temp[ih] = (r*w[:,:,ih]).sum()

    Q = b + temp

    return Q

def LocalEnergy(NumberParticles, Dimension, NumberHidden, r, a, b, w, interaction):

    #sigma=1.0           # From Morten
    #sig2 = sigma**2     # From Morten

    locenergy = 0.0
    kinetic_energy = 0.0
    potential_energy = 0.0

    Q = Qfac(NumberHidden, r, b, w)
    dpsi = 0.0
    dpsi2 = 0.0
    for iq in range(NumberParticles):
        for ix in range(Dimension):
            sum1 = 0.0
            sum2 = 0.0
            for ih in range(NumberHidden):
                sum1 += w[iq,ix,ih]/(1+np.exp(-Q[ih]))
                sum2 += w[iq,ix,ih]**2 * np.exp(Q[ih]) / (1.0 + np.exp(Q[ih]))**2
                #print("ihQ: ", (1+np.exp(-Q[ih])))
                #sum2 += w[iq,ix,ih]**2 * np.exp(-Q[ih]) / (1.0 + np.exp(-Q[ih]))**2 # ???

            #dlnpsi1 = -(r[iq,ix] - a[iq,ix]) /sig2 + sum1/sig2 # From Moren
            #dlnpsi2 = -1/sig2 + sum2/sig2**2                   # From Morten

            dlnpsi1 = -(r[iq, ix] - a[iq, ix]) + sum1
            dlnpsi2 = -1.0 + sum2
            dpsi += -0.5*dlnpsi1*dlnpsi1
            dpsi2 += -0.5*dlnpsi2
            locenergy += 0.5*(-dlnpsi1*dlnpsi1 - dlnpsi2 + r[iq,ix]**2)
            kinetic_energy += -0.5*(dlnpsi1*dlnpsi1 + dlnpsi2)
            potential_energy += 0.5*r[iq,ix]**2
            #print("KE: ", kinetic_energy)
            #print("PE: ", potential_energy)
            #print("gauss: ", -0.5*(r[iq, ix]-a[iq, ix]))
            #print("sum: ", 0.5*sum1)

    if(interaction==True):
        for iq1 in range(NumberParticles):
            for iq2 in range(iq1):
                distance = 0.0
                for ix in range(Dimension):
                    distance += (r[iq1,ix] - r[iq2,ix])**2

                locenergy += 1 / np.sqrt(distance)
    print("Grad : ", dpsi)
    print("Grad2 : ", dpsi2)
    print("Potential: ", potential_energy)
    #locenergy = kinetic_energy + potential_energy
    return locenergy

nsamples = 5000
training_iterations = 100

N = 2
dim = 2
nhidden = 2
M = N*dim
#rng = default_rng(2113)
#r = np.array([[0.30471707975443135, -1.0399841062404955],
r = np.array([0.30471707975443135, -1.0399841062404955,
             0.7504511958064572, 0.9405647163912139])
v_bias = np.array([-0.00631753,  0.01129719, -0.001397, -0.01849913])
h_bias = np.array([0.00869276, -0.00643394])
kernel = np.array([[-0.40775875,  0.08298116],
                   [-0.36875534,  0.03443719],
                   [0.40923255, -0.04661963],
                   [-1.21311022,  0.80609878]])
#jax_wf = NonInteractRBM()



#wf = vmc.RBMWF(N, dim, nhidden, scale=0.5, rng=default_rng, seed=2113)

#print("JAX wf: ", jax_wf.wf(r, wf._a, wf._b, wf._W))
#print("JAX df: ", jax_wf.drift_force(r, wf._a, wf._b, wf._W))
#print("JAX LE: ", jax_wf.local_energy(r, wf._a, wf._b, wf._W))
"""
print("JAX grad a: ", jax_wf.grad_v_bias(r, wf._a, wf._b, wf._W))
print("JAX grad b: ", jax_wf.grad_h_bias(r, wf._a, wf._b, wf._W))
print("JAX grad W: ", jax_wf.grad_kernel(r, wf._a, wf._b, wf._W))
"""
##print("wf df: ", wf.drift_force(r))
#print("wf LE: ", wf.local_energy(r))
"""
print("wf grad a: ", wf.grad_a(r))
print("wf grad b: ", wf.grad_b(r))
print("wf grad W: ", wf.grad_weights(r))
"""
#wf._Q(r)
#wf._denominator(r)
#rng = default_rng(None)
#rbm = vmc.samplers.LMHRBM(wf)
#initial_positions = rng.normal(loc=0.0, scale=1.0, size=(N, dim))
print("LE: ", LocalEnergy(N, dim, nhidden, r.reshape(N, dim), v_bias.reshape(N, dim), h_bias, kernel.reshape(N, dim, nhidden), False))


"""
energies = rbm.train(training_iterations, nsamples, initial_positions, eta=0.1)

for i, energy in enumerate(energies):
    #if energy < 2.5:
    plt.scatter(i, energy)
    plt.hlines(0.5, 0, 100, linestyle="dashed")
    plt.xlabel("Epoch")
    plt.ylabel("Energy")
plt.savefig("test.pdf")
"""
