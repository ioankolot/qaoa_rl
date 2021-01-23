import numpy as np 
from qiskit import QuantumCircuit, execute, Aer, IBMQ, QuantumRegister, ClassicalRegister
from qiskit.compiler import transpile, assemble
import networkx as nx
import math




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
    counts = execute(QAOA, Aer.get_backend('qasm_simulator'), shots=500).result().get_counts() #the default shots are 1024

    return counts  

def get_offset(graph):
    offset = -graph.number_of_edges()/2
    return offset


number_of_qubits = 13
G=nx.random_regular_graph(4, number_of_qubits, seed=10)
print(len(G.edges()))
w = nx.to_numpy_matrix(G, nodelist=sorted(G))

def energies(betas, gammas, number_of_qubits, w, layers, graph):
    qaoa_counts = Qaoa_algorithm(betas, gammas, number_of_qubits, w, layers)
    energies = []

    for sample in list(qaoa_counts.keys()):
        y = [int(num) for num in list(sample)]
        tmp_eng = cost_hamiltonian(y, w, number_of_qubits) + get_offset(graph)
        energies.append(tmp_eng)
    energies.sort(reverse=False)

    return energies


def cvar(alpha, energies):
    len_list = len(energies)
    ak = math.ceil(len_list * alpha)
    cvar = 0
    for sample in range(ak):
        cvar += energies[sample] / ak
    return cvar

