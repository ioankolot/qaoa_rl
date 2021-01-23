from qiskit import IBMQ, execute
from qiskit import QuantumCircuit
from qiskit.providers.aer import QasmSimulator
from qiskit.tools.visualization import plot_histogram
import matplotlib.pyplot as plt

from qiskit.test.mock import FakeVigo
device_backend = FakeVigo()

circ = QuantumCircuit(3, 3)
circ.h(0)
circ.cx(0, 1)
circ.cx(1, 2)
circ.measure([0,1,2], [0,1,2])

ideal_simulator = QasmSimulator()

result = execute(circ, ideal_simulator).result()
counts = result.get_counts(circ)
plot_histogram(counts, title='Ideal counts for 3-qubit GHZ state')
plt.show()


vigo_simulator = QasmSimulator.from_backend(device_backend)

result_noise = execute(circ, vigo_simulator).result()
counts_noise = result_noise.get_counts(circ)
plot_histogram(counts_noise, title="Counts for 3-qubit GHZ state with device noise model")
plt.show()