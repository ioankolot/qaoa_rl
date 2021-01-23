import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
from os import path
import math 
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
import os
from stable_baselines import TD3
from stable_baselines.td3.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.common.callbacks import CheckpointCallback
from stable_baselines.common.callbacks import EvalCallback



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
    counts = execute(QAOA, Aer.get_backend('qasm_simulator'), shots=1000).result().get_counts() #the default shots are 1024

    return counts  


def energies(betas, gammas, number_of_qubits, w, layers, graph):
    qaoa_counts = Qaoa_algorithm(betas, gammas, number_of_qubits, w, layers)
    energies = []

    for sample in list(qaoa_counts.keys()):
        y = [int(num) for num in list(sample)]
        tmp_eng = cost_hamiltonian(y, w, number_of_qubits) + get_offset(graph)
        energies.append(tmp_eng)
    energies.sort(reverse=False)

    return energies


def get_expected_value(betas, gammas, number_of_qubits, w, layers, graph):
    avr_c = 0
    qaoa_counts = Qaoa_algorithm(betas, gammas, number_of_qubits, w, layers)
    for sample in list(qaoa_counts.keys()):
        y = [int(num) for num in list(sample)]
        tmp_eng = cost_hamiltonian(y, w, number_of_qubits) + get_offset(graph)
        avr_c += qaoa_counts[sample] * tmp_eng   
    energy_expectation = avr_c/1000

    return energy_expectation

def cvar(alpha, energies):
    len_list = len(energies)
    ak = math.ceil(len_list * alpha)
    cvar = 0
    for sample in range(ak):
        cvar += energies[sample] / ak
    return cvar

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
w = nx.to_numpy_matrix(G, nodelist=sorted(G))


def get_offset(graph):
    offset = -graph.number_of_edges()/2
    return offset

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

        self.max_layers = 3
        self.time = 0
        self.action3 = None

        self.gammas = []
        self.betas = []

        self.reward_range = (0,1)

        self.action_space = spaces.Box(np.array([-1 for _ in range(2*self.max_layers + 1)]), np.array([1 for __ in range(2*self.max_layers + 1)], dtype = np.float32))
        self.observation_space = spaces.Box(np.array([1, 0, 0, 0, 0, 0, 0,0]), np.array([self.max_layers, np.pi, 2*np.pi, np.pi, 2*np.pi, np.pi, 2*np.pi,20]), dtype = np.float32)
        self.observation_space.shape = (2*self.max_layers + 2,)
        self.state = np.array([1/2 for __ in range(self.number_of_qubits)])

    def _get_reward(self, betas, gammas):
        return -cvar(alpha=0.1, energies = energies(betas, gammas, self.number_of_qubits, self.w, self.layers, self.G))/20 -self._get_noise_penalty() #self._parameter_penalty(betas, gammas)

#    def _get_reward(self, betas, gammas):
#        return -get_expected_value(self.betas, self.gammas, self.number_of_qubits, self.w, self.layers, self.G)/20 #-self._get_noise_penalty() + self._parameter_penalty(betas, gammas)


    def _get_noise_penalty(self):
        return (self.layer_limit)*number_of_gates(self.w)*0.0005
        

    def reset(self):
        self.gammas = [0 for _ in range(self.max_layers)]
        self.betas = [0 for _ in range(self.max_layers)]
        self.actions = [0 for _ in range(self.max_layers)]
        self.action3 = None
        self.action1 = None
        self.time = 0
        self.state = np.array([0 for _ in range(2*self.max_layers+2)])
        self.time_limit = 3
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
                action[0] = np.random.choice([0, 1])

            #action[0] = 2*action[0] + 3
            action[0] = action[0] + 2
            action[0] = round(action[0])
            self.num_of_layers = action[0]
            self.layer_limit = action[0]
            self.time_limit = self.layer_limit
            self.layers = int(action[0])

        action[0] = self.num_of_layers

        self.time += 1

        if self.time == 1:
            action[1] = np.clip(action[1], -1, 1)
            action[2] = np.clip(action[1], -1, 1)
            action[3] = 0
            action[4] = 0
            action[5] = 0
            action[6] = 0
            self.action1 = action[1]
            self.action2 = action[2]
            self.betas[0] = np.pi*action[1]/2 + np.pi/2
            self.gammas[0] = np.pi*action[2] + np.pi

        if self.action1:
            action[1] = self.action1
            action[2] = self.action2
    
        if self.time == 2:
            action[3] = np.clip(action[1], -1, 1)
            action[4] = np.clip(action[1], -1, 1)
            action[5] = 0
            action[6] = 0
            self.action3 = action[3]
            self.action4 = action[4]
            self.betas[1] = np.pi*action[3]/2 + np.pi/2
            self.gammas[1] = np.pi*action[4] + np.pi

        if self.action3:
            action[3] = self.action3
            action[4] = self.action4

        if self.time == 3:
            action[5] = np.clip(action[1], -1, 1)
            action[6] = np.clip(action[1], -1, 1)
            self.betas[2] = np.pi*action[5]/2 + np.pi/2
            self.gammas[2] = np.pi*action[6] + np.pi


        
        reward = self._get_reward(self.betas, self.gammas)
        
        self.state = np.array([self.num_of_layers, self.betas[0], self.gammas[0], self.betas[1], self.gammas[1], self.betas[2], self.gammas[2], reward*20])
        if self.time!= self.time_limit:
            reward = 0
        ob = self.state
        return ob , reward , self.is_done(), {}


env = Qaoa_env()

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
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=0.2 * np.ones(n_actions))

model = TD3(MlpPolicy, env, action_noise=action_noise, verbose=1, gamma=1)
model.learn(total_timesteps=20000, log_interval=10) #callback = eval_callback)
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
