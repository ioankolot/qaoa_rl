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
from qiskit.providers.aer import QasmSimulator
from qiskit.aqua import QuantumInstance
from qiskit.aqua.operators import Z, X, I, StateFn, CircuitStateFn, PauliExpectation, CircuitSampler
import networkx as nx
import matplotlib.pyplot as plt
from   matplotlib import cm
from   matplotlib.ticker import LinearLocator, FormatStrFormatter
import random
import scipy.optimize
import os
from stable_baselines import TD3
from stable_baselines.td3.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.common.callbacks import CheckpointCallback
from stable_baselines.common.callbacks import EvalCallback

from qiskit.test.mock import FakeVigo
device_backend = FakeVigo()

backend = Aer.get_backend('qasm_simulator') # Tell Qiskit how to simulate our circuit
q_instance = QuantumInstance(backend, shots=100)
vigo_simulator = QasmSimulator.from_backend(device_backend)


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




def new_ZZ(q0,q1, gamma, qreg):
    q_encode = QuantumCircuit(qreg)
    q_encode.cx(q0, q1)
    q_encode.u1(gamma, q1)
    q_encode.cx(q0, q1)
    return q_encode


def new_Qaoa_algorithm(betas, gammas, number_of_qubits, w, layers):
    qreg = QuantumRegister(number_of_qubits, name='q')
    QAOA = QuantumCircuit(qreg)

    QAOA.h(range(number_of_qubits))  
    for layer in range(layers):
        for i in range(number_of_qubits):
            for j in range(number_of_qubits):
                if w[i,j] != 0:
                    QAOA += new_ZZ(i,j,gammas[layer],qreg)
        
        for qubit in np.arange(number_of_qubits):
            QAOA.rx(betas[layer], qubit)
       
    state = StateFn(QAOA)
    
    return state

'''
expectation_value = (~psi @ operator @ psi).eval()
print('Math:', psi.adjoint().compose(operator).compose(psi).eval().real)

print(expectation_value.real)  # -1.0
'''


graph_instances = []

for nodes in range(10, 15):
    for seed in range(10):
        for regularity in (3, 6):
            try:
                graph_instances.append(nx.random_regular_graph(regularity, nodes, seed*10))
                graph_instances.append(nx.random_regular_graph(regularity, nodes, (seed+1)*11))
            except:
                pass

def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')



def plot_results(log_folder, title='Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    plt.show()


number_of_qubits = 13

G=nx.random_regular_graph(4, number_of_qubits, seed=10)
print(len(G.edges()))
w = nx.to_numpy_matrix(G, nodelist=sorted(G))

'''
G = graph_instances[150]
number_of_qubits = 13
w = nx.to_numpy_matrix(G, nodelist=sorted(G))


Z_operators = [Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I, I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I, I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I,
I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I, I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I, I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I, 
I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I, I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I, I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I,
I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I, I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I, I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I,
I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z]
X_operators = [X ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I, I ^ X ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I, I ^ I ^ X ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I,
I ^ I ^ I ^ X ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I, I ^ I ^ I ^ I ^ X ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I, I ^ I ^ I ^ I ^ I ^ X ^ I ^ I ^ I ^ I ^ I ^ I ^ I, 
I ^ I ^ I ^ I ^ I ^ I ^ X ^ I ^ I ^ I ^ I ^ I ^ I, I ^ I ^ I ^ I ^ I ^ I ^ I ^ X ^ I ^ I ^ I ^ I ^ I, I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ X ^ I ^ I ^ I ^ I,
I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ X ^ I ^ I ^ I, I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ X ^ I ^ I, I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ X ^ I,
I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ X]


def local_expectations(betas, gammas, number_of_qubits, w, layers):
    expectations = []
    psi = new_Qaoa_algorithm(betas, gammas, number_of_qubits, w, layers)
    for num in range(number_of_qubits):
        measurable_expression = StateFn(Z_operators[num], is_measurement=True).compose(psi)
        expectation = PauliExpectation().convert(measurable_expression)
        sampler = CircuitSampler(q_instance).convert(expectation)
        value = sampler.eval().real
        expectations.append(value)
        measurable_expression = StateFn(X_operators[num], is_measurement=True).compose(psi)
        expectation = PauliExpectation().convert(measurable_expression)
        sampler = CircuitSampler(q_instance).convert(expectation)
        value = sampler.eval().real
        expectations.append(value)
    return np.array(expectations)

'''

def get_offset(graph):
    offset = -graph.number_of_edges()/2
    return offset

def get_expected_value(betas, gammas, number_of_qubits, w, layers, graph):
    avr_c = 0
    qaoa_counts = Qaoa_algorithm(betas, gammas, number_of_qubits, w, layers)[2]
    mean_z = []
    for sample in list(qaoa_counts.keys()):
        y = [int(num) for num in list(sample)]
        mean_z.append(y)
        tmp_eng = cost_hamiltonian(y, w, number_of_qubits) + get_offset(graph)
        avr_c += qaoa_counts[sample] * tmp_eng   
    energy_expectation = avr_c/100
    z_expectation = [0 for _ in range(number_of_qubits)]
    for count in range(len(mean_z)):
        for qubit in range(number_of_qubits):
            z_expectation[qubit] += mean_z[count][qubit]/len(mean_z)
            
    return [energy_expectation, z_expectation]

def number_of_gates(w):
    gates = 0
    for i in range(number_of_qubits):
        for j in range(number_of_qubits):
            if w[i,j] != 0:
                gates +=1
    gates += number_of_qubits
    return gates


class Qaoa_env(gym.Env):
    metadata = {
    'render.modes': ['human', 'rgb_array'],
    'video.frames_per_second': 30
    }
    def __init__(self):
        self.__version__ = '0.1.0'
        

        self.number_of_qubits = number_of_qubits
        self.w = w 
        self.G = G 

        self.max_layers = 5
        self.time = 0

        self.gammas = []
        self.betas = []

        self.reward_range = (0,1)

        self.action_space = spaces.Box(np.array([-1 for _ in range(2*self.max_layers + 1)]), np.array([1 for __ in range(2*self.max_layers + 1)], dtype = np.float32))
        self.observation_space = spaces.Box(np.array([-1 for _ in range(self.number_of_qubits)]), np.array([1 for _ in range(self.number_of_qubits)]) , dtype = np.float32)
        self.observation_space.shape = (self.number_of_qubits,)
        self.state = np.array([1/2 for __ in range(self.number_of_qubits)])


    def _get_observation(self, beta, gamma):
        return get_expected_value(beta, gamma, self.number_of_qubits, self.w, self.layers, self.G)[1]

#    def get_obs(self, betas, gammas, number_of_qubits, w, layers):
#        return np.array(local_expectations(betas, gammas, number_of_qubits, w, layers))

    def _get_reward(self, betas, gammas):
        return -get_expected_value(self.betas, self.gammas, self.number_of_qubits, self.w, self.layers, self.G)[0] #-self._get_noise_penalty() + self._parameter_penalty(betas, gammas)

    def _get_noise_penalty(self):
        return (self.layer_limit)*number_of_gates(self.w)*0.00001
        

    def reset(self):
        self.gammas = []
        self.betas = []
        self.actions = [0 for _ in range(self.max_layers)]
        self.time = 0
        self.state = np.array([1/2 for _ in range(self.number_of_qubits)])
        self.time_limit = 5
        return self.state

    def is_done(self):
        if self.time == self.time_limit:
            return True
        return False

    def step(self, action):
        if self.time == 0:

            #We put some e-greediness to help explore the higher p-values
            action[0] = np.clip(action[0], -1, 1)
            
            if np.random.uniform()<0.15:
                action[0] = np.random.choice([0, 0.5, 1])

            action[0] = 2*action[0] + 3
            #action[0] = action[0] + 2
            action[0] = round(action[0])
            self.num_of_layers = action[0]
            self.layer_limit = action[0]
            self.time_limit = self.layer_limit

        action[0] = self.num_of_layers
    
        self.time += 1
        self.layers = self.time

        if self.time != 0:
            action[int(2*self.time-1)] = np.clip(action[int(2*self.time-1)], -1, 1)
            action[int(2*self.time)] = np.clip(action[int(2*self.time)], -1, 1)
            self.gammas.append(np.pi*action[int(2*self.time)] + np.pi)
            self.betas.append(np.pi*action[int(2*self.time-1)]/2 + np.pi/2)



#        for num in range(int(self.time) + 1, int(self.max_layers+1)):
#            action[2*num] = 0
#            action[2*num-1] = 0



        reward = 0
        if self.time == self.time_limit:
            reward = self._get_reward(self.betas, self.gammas)

        
        self.state = np.array(self._get_observation(self.betas, self.gammas))
        ob = self.state
        #reward = self._get_reward(self.betas, self.gammas)
        return ob , reward , self.is_done(), {}

env = Qaoa_env()
print(env.reset())

print(env.step(env.action_space.sample()))


log_dir = '/tmp/gym/qaoa/'
os.makedirs(log_dir, exist_ok=True)
env = Monitor(env, log_dir)

# Uncomment the lines below to specify which gpu to run on
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#We define the callbacks
#checkpoint_callback = CheckpointCallback(save_freq = 1000, save_path = './logs/', name_prefix='rl_model')
eval_callback = EvalCallback(env, best_model_save_path='./logs/', log_path='./logs/', eval_freq=500, deterministic=True, render=False)

n_actions = env.action_space.shape[-1]
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=0.5 * np.ones(n_actions))

model = TD3(MlpPolicy, env, action_noise=action_noise, verbose=1, gamma=1)
model.learn(total_timesteps=50000, log_interval=10) #callback = eval_callback)
#model.save('rl-qaoa')

#del model

#model = TD3.load("rl-qaoa")


for i in range(20):
    obs = env.reset()
    while not env.is_done():
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        print(action)
        print(obs,rewards)
print(len(G.edges))

plot_results(log_dir)
