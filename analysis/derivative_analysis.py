import sympy as sp
import cupy as cp
import numpy as np

def to_numpy_if_cupy(array):
    if isinstance(array, cp.ndarray):
        return array.get()
    return array

def generate_derivative_analysis(symbolic_expr, weight_matrix, variables):
    weights = to_numpy_if_cupy(weights)
    input_data = to_numpy_if_cupy(input_data)
    weight_matrix_cpu = to_numpy_if_cupy(weight_matrix)

    derivatives = [sp.diff(symbolic_expr, var) for var in variables]
    sample_point = {var: 1 for var in variables}
    derivative_vals = [float(d.evalf(subs=sample_point)) for d in derivatives]

    weights_first_neuron = weight_matrix_cpu[0]
    deviations = weights_first_neuron - np.array(derivative_vals)

    analysis = {
        "symbolic_derivatives": derivatives,
        "derivative_values_at_1": derivative_vals,
        "weights_first_neuron": weights_first_neuron,
        "deviations": deviations,
        "mean_abs_deviation": np.mean(np.abs(deviations))
    }
    return analysis