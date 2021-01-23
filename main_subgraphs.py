import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
from os import path
import math 
from typing import Any, Dict, List, Tuple
from collections import deque
import pkg_resources
from qiskit import QuantumCircuit, execute, Aer, IBMQ, QuantumRegister, ClassicalRegister
from qiskit.compiler import transpile, assemble
import networkx as nx
import matplotlib.pyplot as plt
from   matplotlib import cm
from   matplotlib.ticker import LinearLocator, FormatStrFormatter
import random
import scipy.optimize

from stable_baselines import TD3
from stable_baselines.td3.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

#Now for the QAOA. First of all, we define the ZZ gate. The below gate is up to a global phase the same with e^(igZZ).

def ZZ(q0,q1, gamma,qreg,creg):
    q_encode = QuantumCircuit(qreg,creg)
    q_encode.cx(q0, q1)
    q_encode.u1(gamma, q1)
    q_encode.cx(q0, q1)
    return q_encode

#We define the sigma(z) pauli operator

def sigma(z):
    if z == 0:
        value = 1
    elif z == 1:
        value = -1
    return value

#There exists bijective map between the cost function and the interaction Hamiltonian. The function 
#below acquires a bitstring as an input and outputs the total energy. An important note is that in qiskit for a state
#eg |11110> , the 0th qubit is in the 0 state. That is we wright it in the top right position.

def cost_hamiltonian(x, w, number_of_qubits):
    spins = []
    for i in x[::-1]: #reverses the bitstring
        spins.append(int(i))
    total_energy = 0
    for i in range(number_of_qubits):
        for j in range(number_of_qubits):
            if w[i,j] != 0:
                total_energy += sigma(spins[i])*sigma(spins[j])
    total_energy /= 4
    return total_energy

#In the last step, we repeat the expirement many times and measure in the computational basis. We use
#the most common measurement to calculate the total energy and save the bitstring. Probably it is more convenient
#to define a class and call qaoa.circuit() for example to draw the circuit but it is left for future work.

def Qaoa_algorithm(betas, gammas, number_of_qubits, w, layers):
    qreg = QuantumRegister(number_of_qubits, name='q')
    creg = ClassicalRegister(number_of_qubits, name = 'c')
    QAOA = QuantumCircuit(qreg, creg)
    
    QAOA.h(range(number_of_qubits))
    QAOA.barrier()
    
    for layer in range(layers):
        for i in range(number_of_qubits):
            for j in range(number_of_qubits):
                if w[i,j] != 0:
                    QAOA += ZZ(i,j,gammas[layer],qreg, creg)
        QAOA.barrier()
        for qubit in np.arange(number_of_qubits):
            QAOA.rx(betas[layer], qubit)
    QAOA.barrier()
    
    QAOA.measure(range(number_of_qubits), creg)
    counts = execute(QAOA, Aer.get_backend('qasm_simulator'), shots=100).result().get_counts() #the default shots are 1024
    max_value = max(counts, key=counts.get)  #gives the most probable bitstring
    energy = cost_hamiltonian(max_value, w, number_of_qubits)
    
    list = []
    list.append(float(energy))
    list.append(max_value)
    list.append(counts)
    return list  

graph_instances = []

for nodes in range(12, 16):
    for seed in range(10):
        for regularity in [3]:
            try:
                graph_instances.append(nx.random_regular_graph(regularity, nodes, seed*10))
                graph_instances.append(nx.random_regular_graph(regularity, nodes, (seed+1)*11))
            except:
                pass

graph = graph_instances[1]
subgraphs = []
subgraph1 = graph.subgraph([0, 4, 5, 9])
subgraph2 = graph.subgraph([3, 6, 8, 10])
nodelist = []
for node in graph.nodes():
    if node not in subgraph1.nodes() and node not in subgraph2.nodes():
        nodelist.append(node)

subgraph3 = graph.subgraph(nodelist)


subgraphs.append(subgraph1)
subgraphs.append(subgraph2)
subgraphs.append(subgraph3)

number_of_qubits = len(graph.nodes())
w = nx.to_numpy_matrix(graph, nodelist=sorted(graph))

colors = ['r' for node in graph.nodes()]
default_axes = plt.axes(frameon=True)
pos = nx.spring_layout(graph)
nx.draw_networkx(graph, node_color = colors, node_size = 600, alpha = 1, ax=default_axes, pos=pos)
plt.show()



best_cost = 0
for b in range(2**number_of_qubits):
    x = [int(t) for t in reversed(list(bin(b)[2:].zfill(number_of_qubits)))]
    cost = 0
    for i in range(number_of_qubits):
        for j in range(number_of_qubits):
                cost += w[i,j] * x[i] * (1-x[j])
    if best_cost < cost:
        best_cost = cost

print(best_cost)
print(len(graph.edges()))

def get_offset(graph):
    offset = -graph.number_of_edges()/2
    return offset

def get_expected_value(betas, gammas, number_of_qubits, w, layers, graph):
    avr_c = 0
    qaoa_counts = Qaoa_algorithm(betas, gammas, number_of_qubits, w, layers)[2]
    for sample in list(qaoa_counts.keys()):
        y = [int(num) for num in list(sample)]
        tmp_eng = cost_hamiltonian(y, w, number_of_qubits) + get_offset(graph)
        avr_c += qaoa_counts[sample] * tmp_eng   
    energy_expectation = avr_c/100 
    return energy_expectation

def number_of_gates(w):
    gates = 0
    for i in range(number_of_qubits):
        for j in range(number_of_qubits):
            if w[i,j] != 0:
                gates +=1
    gates += number_of_qubits
    return gates

print(number_of_gates(w))    

class Qaoa_env():
    def __init__(self):
        self.__version__ = '0.1.0'

        self.number_of_qubits = number_of_qubits
        self.w = w 
        self.G = graph 

        self.max_layers = 3
        self.time = 0

        self.gammas = []
        self.betas = []

        self.action_space = spaces.Box(np.array([-1 for _ in range(2*self.max_layers + 1)]), np.array([1 for __ in range(2*self.max_layers + 1)], dtype = np.float32))
        self.observation_space = spaces.Box(np.array([-1 for _ in range(self.number_of_qubits)]), np.array([1 for _ in range(self.number_of_qubits)]) , dtype = np.float32)
        self.observation_space.shape = (self.number_of_qubits,)
        self.state = np.array([0 for __ in range(self.number_of_qubits)])


    def _get_observation(self, beta, gamma):
        return get_expected_value(beta, gamma, self.number_of_qubits, self.w, self.layers, self.G)

    def _get_reward(self, betas, gammas):
        return -get_expected_value(self.betas, self.gammas, self.number_of_qubits, self.w, self.layers, self.G)/18 #-self._get_noise_penalty()# + self._parameter_penalty(betas, gammas)

    def _get_noise_penalty(self):
        return (self.layer_limit)*number_of_gates(self.w)*0.001
        

    def reset(self):
        self.gammas = []
        self.betas = []
        self.time = 0
        self.state = np.array([0 for _ in range(self.number_of_qubits)])
        self.time_limit = 3
        return self.state

    def is_done(self):
        if self.time == self.time_limit:
            return True
        return False

    def step(self, action):
        if self.time == 0:
            #put some e-greediness
            action[0] = np.clip(action[0], -1, 1)
            if np.random.uniform()<0.15:
                action[0] = np.random.choice([0,1])
            action[0] = action[0] + 2
 #           action[0] = 2*action[0] + 3
            action[0] = round(action[0])
            self.kol = action[0]
            self.layer_limit = action[0]
            self.time_limit = self.layer_limit

        action[0] = self.kol
    
        self.time += 1
        self.layers = self.time

        

        if self.time != 0:
            action[int(2*self.time-1)] = np.clip(action[int(2*self.time-1)], -1, 1)
            action[int(2*self.time)] = np.clip(action[int(2*self.time)], -1, 1)
            self.gammas.append(np.pi*action[int(2*self.time)] + np.pi)
            self.betas.append(np.pi*action[int(2*self.time-1)]/2 + np.pi/2)

        


        for num in range(int(self.time) + 1, int(self.max_layers+1)):
            action[2*num] = 0
            action[2*num-1] = 0

#        reward = 0
#        if self.time == self.time_limit:
#            reward = self._get_reward(self.betas, self.gammas)


        self.state = np.asarray(list(Qaoa_algorithm(self.betas, self.gammas, self.number_of_qubits, self.w, self.layers)[1]), dtype=np.float32)
        ob = self.state
        reward = self._get_reward(self.betas, self.gammas)
        return ob , reward/self.time_limit , self.is_done(), {}


'''
number_of_qubits = 13
G=nx.random_regular_graph(4, number_of_qubits, seed=10)
w = nx.to_numpy_matrix(G, nodelist=sorted(G))


for i,graph in enumerate(graph_instances[20:30]):
    print('Graph: ', i)
    colors = ['r' for node in graph.nodes()]
    default_axes = plt.axes(frameon=True)
    pos = nx.spring_layout(graph)

    nx.draw_networkx(graph, node_color=colors, node_size=600, alpha=1, ax=default_axes, pos=pos)
    plt.show()

'''