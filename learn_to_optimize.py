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
    energy_expectation = avr_c/100

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
graph_list = [nx.random_regular_graph(4, 12, seed=10), nx.random_regular_graph(4, 12, seed=11),nx.random_regular_graph(4, 12, seed=12),
nx.random_regular_graph(4, 12, seed=13), nx.random_regular_graph(3, 12, seed=10), nx.random_regular_graph(3, 12, seed=11),
nx.random_regular_graph(3, 12, seed=12), nx.random_regular_graph(3, 12, seed=13)]

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

        self.L = 5 #history iterations

        self.layers = 1
        self.max_layers = 1
        self.time = 0

        self.gammas = []
        self.betas = []
        self.reward_range=(-2,3)

        self.time_limit = 64

        self.action_space = spaces.Box(np.array([-1 for _ in range(2*self.layers)]), np.array([1 for __ in range(2*self.layers)]), dtype = np.float32)
        self.observation_space = spaces.Box(np.array([[-20,-100,-100] for _ in range(self.L)]), np.array([[20, 100, 100] for _ in range(self.L)]), dtype = np.float32)

    def _get_observation(self, beta, gamma):
        return -get_expected_value(beta, gamma, self.number_of_qubits, self.w, self.layers, self.G)

    def _get_reward(self, betas, gammas):
        return -get_expected_value(self.betas, self.gammas, self.number_of_qubits, self.w, self.layers, self.G) #-self._get_noise_penalty() + self._parameter_penalty(betas, gammas)

    def reset(self):
        self.random = np.random.choice(range(4))
        self.G = graph_list[self.random]
        self.w = nx.to_numpy_matrix(self.G, nodelist=sorted(self.G))
        self.number_of_qubits = len(self.G.nodes())
        self.gammas = [np.random.uniform(0, 2*np.pi) for _ in range(self.layers)]
        self.betas = [np.random.uniform(0, np.pi) for _ in range(self.layers)]
        self.expectation_value = self._get_observation(self.betas, self.gammas)
        self.time = 0

        self.history1 = np.array([[0, 0, 0]])
        self.history2 = np.array([(3)*[0]])
        self.history3 = np.array([(3)*[0]])
        self.history4 = np.array([(3)*[0]])
        self.history5 = np.array([(3)*[0]])
#        self.history6 = np.array([(2*self.layers+3)*[0]])
#        self.history7 = np.array([(2*self.layers+3)*[0]])
#        self.history8 = np.array([(2*self.layers+3)*[0]])
#        self.history9 = np.array([(2*self.layers+3)*[0]])
#        self.history10 = np.array([(2*self.layers+3)*[0]])
        self.state = np.concatenate((self.history1,self.history2,self.history3,self.history4, self.history5)) #self.history5,self.history6,self.history7,self.history8, self.history9,self.history10))
        return self.state

    def is_done(self):
        if self.time == self.time_limit:
            return True
        return False

    def step(self, action):
#        self.history10 = self.history9
#        self.history9 = self.history8
#        self.history8 = self.history7
#        self.history7 = self.history6
#        self.history6 = self.history5
        self.history5 = self.history4
        self.history4 = self.history3
        self.history3 = self.history2
        self.history2 = self.history1

        self.betas[0] += action[0]
        self.gammas[0] += action[1]
        self.betas[0] = self.betas[0] % np.pi
        self.gammas[0] = self.gammas[0] % np.pi
        self.time += 1

        self.expectation_value_new = self._get_observation(self.betas, self.gammas)
        self.df = self.expectation_value_new - self.expectation_value
        # We use the parameter shift to calculate the gradient. First of all the gradient of betas:
        self.beta_gradient = (self._get_observation(np.array(self.betas) + np.pi/2, self.gammas) - self._get_observation(np.array(self.betas)-np.pi/2, self.gammas))/2
        self.gamma_gradient = (self._get_observation(self.betas, np.array(self.gammas) + np.pi/2)- self._get_observation(self.betas, np.array(self.gammas)-np.pi/2))/2

#        self.history1 = [self.betas[0], self.gammas[0]]
        self.history1 = []
        self.history1.append(self.df)
        self.history1.append(self.beta_gradient)
        self.history1.append(self.gamma_gradient)
#        self.history1.append(self.df/(action[0]*action[1]))
        self.history1 = np.array([self.history1])

        self.expectation_value = self.expectation_value_new

        reward = self.df
        self.state = np.concatenate((self.history1,self.history2,self.history3,self.history4, self.history5)) #,self.history6,self.history7,self.history8, self.history9,self.history10))
        ob = self.state
        return ob , reward, self.is_done(), {}


env = Qaoa_env()
#print(env.reset())
#for i in range(64):
#    print(env.step(env.action_space.sample()))



log_dir = '/tmp/gym/qaoa/'
os.makedirs(log_dir, exist_ok=True)
env = Monitor(env, log_dir)

# Uncomment the lines below to specify which gpu to run on
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#We define the callbacks
#checkpoint_callback = CheckpointCallback(save_freq = 1000, save_path = './logs/', name_prefix='rl_model')
eval_callback = EvalCallback(env, best_model_save_path='./logs/', log_path='./logs/', eval_freq=500, deterministic=True, render=False)
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./logs/',
                                         name_prefix='4_history_rl_model')


n_actions = env.action_space.shape[-1]
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = TD3(MlpPolicy, env, action_noise=action_noise, verbose=1, gamma=0.99)
model.learn(total_timesteps=500000, log_interval=10, callback=checkpoint_callback) #callback = eval_callback)


#model.save('rl-qaoa')

#del model
'''
#model = TD3.load("rl-qaoa")
model = TD3.load('logs/rl_model_250000_steps')
kol = False
for i in range(1):
    obs = env.reset()
    while not env.is_done():
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        print(env.state)
        print(env.expectation_value)
        print(env.betas)
        print(action)
'''


plot_results(log_dir)
