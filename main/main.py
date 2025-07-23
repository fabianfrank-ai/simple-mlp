import cupy as cp
import numpy as np
import matplotlib
import matplotlib.widgets as widgets
import tkinter as tk
from tkinter import ttk, messagebox
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sympy as sp
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D
import threading
from symbolic_backpropagation import symbolicMLP
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import streamlit as st
import time

scaler=StandardScaler()
# Customize matplotlib colors
matplotlib.rcParams.update({
    'figure.facecolor': '#1e1e1e',
    'axes.facecolor': '#2c2c2c',
    'axes.edgecolor': 'white',
    'axes.labelcolor': 'white',
    'axes.titlecolor': 'white',
    'xtick.color': 'lightgray',
    'ytick.color': 'lightgray',
    'text.color': 'white',
    'grid.color': '#444444',
    'legend.facecolor': '#2c2c2c',
    'legend.edgecolor': 'white',
    'savefig.facecolor': '#1e1e1e',
    'savefig.edgecolor': '#1e1e1e',
})

# Check for GPU and set numpy module
if cp.cuda.runtime.getDeviceCount() > 0:
    np = cp
    print("GPU is available, training with cupy")
else:
    np = np
    print("Warning! No Cuda-Version found! Will continue training with CPU")

# Plot Setup
fig = plt.figure(figsize=(16, 9))
gs = GridSpec(3, 4, height_ratios=[4, 2, 0.5])
ax = fig.add_subplot(gs[0, 0:2], projection='3d')
ax2d = fig.add_subplot(gs[0, 2:4])
ax2 = fig.add_subplot(gs[1, 2:4])
ax3 = fig.add_subplot(gs[1, 0:2])
ax.view_init(elev=30, azim=135)

ax.grid(True, linestyle='--', alpha=0.5)
ax2d.grid(True, linestyle='--', alpha=0.5)
ax2.grid(True, linestyle='--', alpha=0.5)
ax3.grid(True, linestyle='--', alpha=0.5)

ax.set_zlabel('y', fontsize=14)
ax.set_title("Training in Action", fontsize=16)
ax2.set_title("Loss, Validation Loss and Weights", fontsize=16)
ax3.set_title("Prediction History", fontsize=16)
ax4 = ax2.twinx()

ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax4.set_ylabel('Weights')
ax3.set_xlabel('Error')
ax3.set_ylabel('Frequency')

st.title("Symbolic MLP Training Dashboard")
st.sidebar.header("Hyperparameters")

with st.sidebar:
    with st.expander("Training Hyperparameters", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.write("Learning Rate ")
            lr = st.slider("", 0.001, 1.0, 0.01, key="lr_slider",help="Controls how much the model's weights are adjusted during training. A smaller learning rate leads to slower but more stable convergence; a larger one trains faster but may overshoot or diverge.")
        with col2:
            st.write("Batch Size ")
            batch_size = st.slider("", 1, 128, 30, step=1, key="batch_slider",help="Defines how many training samples are processed before the model updates its weights. Smaller batches can lead to more noise (but better generalization); larger batches are more stable but need more memory.")

with st.sidebar:
    with st.expander("Regularization & Architecture", expanded=True):
        col3, col4 = st.columns(2)
        with col3:
            st.write("L2 Lambda")
            lambda_l2 = st.slider("", 0.0, 1.0, 0.001, key="lambda_slider",help="Adds a penalty to large weights in order to reduce overfitting. A higher lambda forces the model to keep weights smaller, improving generalization but possibly underfitting.")
        with col4:
            st.write("Hidden Layers")
            hidden_layers = [st.slider("", 1, 300, 50, step=1, key="hl_slider",help="Specifies the number of neurons or layers between input and output. More hidden units or layers allow the model to learn complex patterns, but also increase the risk of overfitting and training time.")]

with st.sidebar.expander("Solver", expanded=False):
    st.write("Optimization algorithm (Adam, SGD, RMSprop).")
    solver = st.radio("", ('Adam', 'SGD', 'RMSprop'), help="Adam= adaptive moment estimation, SGD= stochastic gradient descent, RMSprop= root mean square propagation.",key="solver_radio")

with st.sidebar.expander("Activation", expanded=False):
    st.write("Activation function (tanh, relu, sigmoid, sin, cos, exp).")
    activation = st.radio("", ('tanh', 'relu', 'sigmoid', 'sin', 'cos', 'exp'), key="activation_radio",help="Non-linearity introduces the ability to learn complex patterns.")


# SymPy symbols and allowed functions (move this up)
x1_sym, x2_sym, x3_sym = sp.symbols('x1 x2 x3')
w1_sym, w2_sym, w3_sym, w4_sym, w5_sym, w6_sym, w7_sym, w8_sym, w9_sym, w10_sym, w11_sym, w12_sym, w13_sym, w14_sym, w15_sym, w16_sym, b_sym = sp.symbols('w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12 w13 w14 w15 w16 b')
allowed_functions = {
    'sin': sp.sin,
    'cos': sp.cos,
    'exp': sp.exp,
    'log': sp.log,
    'sqrt': sp.sqrt,
    'pi': sp.pi,
}
w_cos = {x1_sym: w5_sym, x2_sym: w6_sym, x3_sym: w7_sym}
w_sin = {x1_sym: w8_sym, x2_sym: w9_sym, x3_sym: w10_sym}
w_tanh = {x1_sym: w11_sym, x2_sym: w12_sym, x3_sym: w13_sym}
w_exp = {x1_sym: w14_sym, x2_sym: w15_sym, x3_sym: w16_sym}

#dummy nonlinear terms for now, they will be replaced with actual ones later
dummy_nonlinear_terms = {sp.tanh: {x1_sym: sp.Symbol('w_tanh_x1'), x2_sym: sp.Symbol('w_tanh_x2'), x3_sym: sp.Symbol('w_tanh_x3')}}
mlp = symbolicMLP(np, dummy_nonlinear_terms, 0.001, 0.01, [50])
# Update MLP based on selections
if solver == 'Adam':
    mlp.solver = 'adam'
    print("Using Adam optimizer")
elif solver == 'SGD':
    mlp.solver = 'sgd'
    print("Using SGD optimizer")
elif solver == 'RMSprop':
    mlp.solver = 'rmsprop'
    print("Using RMSprop optimizer")

if activation == 'tanh':
    mlp.output_activation = sp.tanh
    print("Using tanh activation")
elif activation == 'relu':
    mlp.output_activation = 'relu'
    print("Using relu activation")
elif activation == 'sigmoid':
    mlp.output_activation = 'sigmoid'
    print("Using sigmoid activation")
elif activation == 'sin':
    mlp.output_activation = sp.sin
    print("Using sin activation")
elif activation == 'cos':
    mlp.output_activation = sp.cos
    print("Using cos activation")
elif activation == 'exp':
    mlp.output_activation = sp.exp
    print("Using exp activation")

# User input and visualization variables
user_input = st.text_input("Enter a function in terms of x1, x2, x3 (e.g., 'x1 + x2 + x3 + 2', 'sin(x3) + cos(x2)'): ")
vis_vars = st.text_input("Choose two visualization variables (e.g., x1,x2 or x1,x3) [default: x1,x2]")

# SymPy symbols and allowed functions
x1_sym, x2_sym, x3_sym = sp.symbols('x1 x2 x3')
w1_sym, w2_sym, w3_sym, w4_sym, w5_sym, w6_sym, w7_sym, w8_sym, w9_sym, w10_sym, w11_sym, w12_sym, w13_sym, w14_sym, w15_sym, w16_sym, b_sym = sp.symbols('w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12 w13 w14 w15 w16 b')
allowed_functions = {
    'sin': sp.sin,
    'cos': sp.cos,
    'exp': sp.exp,
    'log': sp.log,
    'sqrt': sp.sqrt,
    'pi': sp.pi,
}
w_cos = {x1_sym: w5_sym, x2_sym: w6_sym, x3_sym: w7_sym}
w_sin = {x1_sym: w8_sym, x2_sym: w9_sym, x3_sym: w10_sym}
w_tanh = {x1_sym: w11_sym, x2_sym: w12_sym, x3_sym: w13_sym}
w_exp = {x1_sym: w14_sym, x2_sym: w15_sym, x3_sym: w16_sym}

def parse_expression(user_input):
    try:
        if not user_input or not user_input.strip():
            print("Invalid or empty input. Using default function: x1 + x2 + x3 + 1")
            user_input = "x1 + x2 + x3 + 1"
        
        symbol_dict = {str(s): s for s in [x1_sym, x2_sym, x3_sym, w1_sym, w2_sym, w3_sym, b_sym]}
        symbol_dict.update(allowed_functions)
        
        expr = sp.sympify(user_input, locals=symbol_dict, evaluate=False)

        nonlinear_terms = {
            sp.sin: {},
            sp.cos: {},
            sp.exp: {},
            sp.tanh: {}
        }

        for func, weight_dict in nonlinear_terms.items():
            for term in expr.atoms(func):
                variable = term.args[0]
                if variable in [x1_sym, x2_sym, x3_sym]:
                    weight_symbol = sp.Symbol(f"w_{func.__name__}_{variable}")
                    weight_dict[variable] = weight_symbol

        w1_true = expr.diff(x1_sym) if x1_sym in expr.free_symbols else 0
        w2_true = expr.diff(x2_sym) if x2_sym in expr.free_symbols else 0
        w3_true = expr.diff(x3_sym) if x3_sym in expr.free_symbols else 0

        _, b_true = expr.as_independent(x1_sym, x2_sym, x3_sym)

        expr_train = w1_sym * x1_sym + w2_sym * x2_sym + w3_sym * x3_sym + b_sym
        for func, weight_dict in nonlinear_terms.items():
            for var, weight in weight_dict.items():
                expr_train += weight * func(var)

        all_weights = list({
            *nonlinear_terms[sp.sin].values(),
            *nonlinear_terms[sp.cos].values(),
            *nonlinear_terms[sp.exp].values(),
            *nonlinear_terms[sp.tanh].values(),
            w1_sym, w2_sym, w3_sym, b_sym
        })

        f_true = sp.lambdify([x1_sym, x2_sym, x3_sym], expr, modules=np)
        f_train = sp.lambdify([x1_sym, x2_sym, x3_sym] + all_weights, expr_train, modules=np)

        return f_train, f_true, w1_true, w2_true, w3_true, b_true, nonlinear_terms, all_weights

    except Exception as e:
        print("Error:", e)
        print("Using default function: x1 + x2 + x3 + 1")
        return parse_expression("x1 + x2 + x3 + 1")

f_train, f_true, w1_true, w2_true, w3_true, b_true, nonlinear_terms, all_weights = parse_expression(user_input)


weight_values = [0] * len(all_weights)

mlp = symbolicMLP(np, nonlinear_terms, lambda_l2, lr, hidden_layers)
mlp_reference = symbolicMLP(np, nonlinear_terms, 0.001, 0.01, [50])

if vis_vars not in ['x1,x2', 'x2,x3', 'x1,x3']:
    vis_vars = 'x1,x2'
    print("Invalid Input. Using default: x1,x2")

# Generate independent random data for x1, x2, x3
np.random.seed(42)
x1 = np.linspace(0, 2 * np.pi, 100)
x2 = np.linspace(0, 1, 100)
x3 = np.zeros_like(x1)
x1_cpu = x1.get()
x2_cpu = x2.get()
x3_cpu = x3.get()

X = np.stack([x1, x2, x3], axis=1)

# Generate true output with noise
y = f_true(x1, x2, x3) + np.random.randn(100) * 0.1

# Split data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
x1_train, x2_train, x3_train = x_train[:, 0], x_train[:, 1], x_train[:, 2]
x1_test, x2_test, x3_test = x_test[:, 0], x_test[:, 1], x_test[:, 2]

X_mean = np.mean(x_train, axis=0)
X_std = np.std(x_train, axis=0) + 1e-8
x1 = (x1 - X_mean[0]) / X_std[0]
x2 = (x2 - X_mean[1]) / X_std[1]
x3 = (x3 - X_mean[2]) / X_std[2]

# Sort for plotting
sort_idx = np.argsort(x1_train)
x1_sorted = x1_train[sort_idx]
x2_sorted = x2_train[sort_idx]
x3_sorted = x3_train[sort_idx]
y_train_sorted = y_train[sort_idx]
y_true_sorted = f_true(x1_sorted, x2_sorted, x3_sorted)

# Initial prediction
y_pred_sorted = f_train(x1_sorted, x2_sorted, x3_sorted, *weight_values)

x1_plot = np.linspace(np.min(x1_train), np.max(x1_train), 200)
x2_plot = np.linspace(np.min(x2_train), np.max(x2_train), 200)
x3_plot = np.linspace(np.min(x3_train), np.max(x3_train), 200)
y_true_plot = f_true(x1_plot, x2_plot, x3_plot)
y_pred_plot = f_train(x1_plot, x2_plot, x3_plot, *weight_values)

# Convert to CPU for Matplotlib
x1_train_cpu = x1_train.get()
x2_train_cpu = x2_train.get()
x3_train_cpu = x3_train.get()
y_train_cpu = y_train.get()
x1_test_cpu = x1_test.get()
x2_test_cpu = x2_test.get()
x3_test_cpu = x3_test.get()
y_test_cpu = y_test.get()
x1_plot_cpu = x1_plot.get()
x2_plot_cpu = x2_plot.get()
x3_plot_cpu = x3_plot.get()
y_true_plot_cpu = y_true_plot.get()
y_pred_plot_cpu = y_pred_plot.get()

# Plotting elements
scatter = ax.scatter(x1_train_cpu, x2_train_cpu, y_train_cpu, label="Training Data", color='blue', marker='o')
scatter_2d = ax2d.scatter([], [], label="Training Data", color='blue', marker='o')
scatter_test = ax.scatter(x1_test_cpu, x2_test_cpu, y_test_cpu, label="Test Data", color='cyan', marker='x')
scatter_test_2d = ax2d.scatter([], [], label="Test Data", color='cyan', marker='x')
scatter_pred = ax.scatter(x1_plot_cpu, x2_plot_cpu, y_pred_plot_cpu)

twod_pred = ax2d.plot([], [])
twod_true = ax2d.plot([], [])

if vis_vars == 'x1,x2':
    x1_mesh, x2_mesh = np.meshgrid(np.linspace(np.min(x1_train_cpu), np.max(x1_train_cpu), 20),
                                   np.linspace(np.min(x2_train_cpu), np.max(x2_train_cpu), 20))
    x3_mesh = np.full_like(x1_mesh, np.mean(x3_train_cpu))
    ax.set_xlabel('x1', fontsize=14)
    ax.set_ylabel('x2', fontsize=14)
elif vis_vars == 'x1,x3':
    x1_mesh, x3_mesh = np.meshgrid(np.linspace(np.min(x1_train_cpu), np.max(x1_train_cpu), 20),
                                   np.linspace(np.min(x2_train_cpu), np.max(x2_train_cpu), 20))
    x2_mesh = np.full_like(x1_mesh, np.mean(x2_train_cpu))
    ax.set_xlabel('x1', fontsize=14)
    ax.set_ylabel('x3', fontsize=14)
elif vis_vars == 'x2,x3':
    x2_mesh, x3_mesh = np.meshgrid(np.linspace(np.min(x1_train_cpu), np.max(x1_train_cpu), 20),
                                   np.linspace(np.min(x2_train_cpu), np.max(x2_train_cpu), 20))
    x1_mesh = np.full_like(x2_mesh, np.mean(x1_train_cpu))
    ax.set_xlabel('x2', fontsize=14)
    ax.set_ylabel('x3', fontsize=14)

mlp_sklearn = MLPRegressor(hidden_layer_sizes=(200,), max_iter=5000, learning_rate_init=0.001, random_state=42, batch_size=580, warm_start=True)
X_train = np.column_stack((x1_mesh.ravel(), x2_mesh.ravel(), x3_mesh.ravel()))
y_train_mesh = f_train(x1_mesh, x2_mesh, x3_mesh, *weight_values).ravel()
X_train_scaled = scaler.fit_transform(X_train.get())
X_train_np = X_train.get()
y_train_mesh_np = y_train_mesh.get()
X_train_scaled = scaler.fit_transform(X_train_np)
mlp_sklearn.fit(X_train_scaled, y_train_mesh_np)

def compute_pred_mesh(mlp):
    global y_pred_mesh, y_pred_sklearn_mesh
    x1_flat, x2_flat, x3_flat = x1_mesh.ravel(), x2_mesh.ravel(), x3_mesh.ravel()
    y_pred_flat = mlp.forward(x1_flat, x2_flat, x3_flat)
    y_pred_mesh = y_pred_flat.reshape(x1_mesh.shape)
    y_pred_sklearn_flat = mlp_sklearn.predict(X_train.get())
    y_pred_sklearn_mesh = y_pred_sklearn_flat.reshape(x1_mesh.shape)
    return y_pred_mesh, y_pred_sklearn_mesh

y_pred_mesh, y_pred_sklearn_mesh = compute_pred_mesh(mlp)
y_true_mesh = f_true(x1_mesh, x2_mesh, x3_mesh)

ax.plot([], [], [], color='white', alpha=0.3, label='Sklearn Prediction')
surface_first_pred = ax.plot_surface(x1_mesh.get(), x2_mesh.get(), y_pred_mesh.get(), cmap='viridis', alpha=0.3, label="First Prediction")
loss_line, = ax2.plot([], [], color='purple', label='Loss')
loss_fix_line, = ax2.plot([], [], color='palegreen', label='Loss with fixed parameters')
val_line, = ax2.plot([], [], color='orange', label='Validation Loss')
sklearn_line = ax2d.plot([], [], color='white', label="sklearn prediction", linestyle='--')
sklearn_mse_line, = ax2.plot([], [], color='black', label='Sklearn MSE')
sklearn_val_line, = ax2.plot([], [], color='gray', label='Sklearn Val Loss')
w1_line, = ax4.plot([], [], label=f"W1: {mlp.weights[0][0, 0]:.4f}", color='green', linestyle='--', alpha=0.3)
w2_line, = ax4.plot([], [], label=f"W2: {mlp.weights[0][0, 1]:.4f}", color='red', linestyle='--', alpha=0.3)
w3_line, = ax4.plot([], [], label=f"W3: {mlp.weights[0][0, 2]:.4f}", color='blue', linestyle='--', alpha=0.3)
b_line, = ax4.plot([], [], label=f"Bias: {mlp.biases[0][0, 0]:.4f}", color='black', linestyle='--', alpha=0.3)

ax2.legend(loc='upper right')
ax4.legend(loc='upper left')
ax.legend(loc='upper left')

contour_true = ax2d.contourf(x1_mesh.get(), x2_mesh.get(), y_true_mesh.get(), cmap='inferno', alpha=0.2, label="true")
contour_pred = ax2d.contourf(x1_mesh.get(), x2_mesh.get(), y_pred_mesh.get(), cmap='viridis', alpha=0.2, label='Model Prediction')
contour_sk = ax2d.contourf(x1_mesh.get(), x2_mesh.get(), y_pred_sklearn_mesh, cmap='plasma', alpha=0.2, label="Sklearn Prediction")
cbar = plt.colorbar(contour_pred, ax=ax2d)
abar = plt.colorbar(contour_true, ax=ax2d)
bbar = plt.colorbar(contour_sk, ax=ax2d)
cbar.set_label("Prediction")
abar.set_label("True Value")
bbar.set_label("Sklearn Prediction")

def to_cpu(arr):
    return arr.get() if isinstance(arr, cp.ndarray) else arr

model_list = [mlp, symbolicMLP]

# Update function for animation
plotted_surfaces = {'pred': None, 'sk': None, 'true': None}

# Initialize lists for storing history
loss_values = []
loss_values_fix = []
val_loss_values = []
epoch_values = []
w1_history = []
w2_history = []
w3_history = []
b_history = []
mlp_sklearn_mse = []
mlp_sklearn_val_loss = []

def update(epoch):
    global loss_values, val_loss_values, epoch_values, w1_history, w2_history, w3_history, b_history
    global mlp_sklearn_val_loss, mlp_sklearn_mse
    lr = st.session_state.lr_slider
    batch_size = st.session_state.batch_slider
    x1_flat, x2_flat, x3_flat = x1_mesh.ravel(), x2_mesh.ravel(), x3_mesh.ravel()
    y_true_flat = f_true(x1_flat, x2_flat, x3_flat)
    mse_train = 0.0
    mse_fix = 0.0
    y_pred_mesh, y_pred_sklearn_mesh = compute_pred_mesh(mlp)
    # Train symbolicMLP in batches
    for i in range(0, len(x1_train), batch_size):
        batch_x1 = x1_train[i:i + batch_size]
        batch_x2 = x2_train[i:i + batch_size]
        batch_x3 = x3_train[i:i + batch_size]
        batch_y_true = y_train[i:i + batch_size]
        y_pred_batch = mlp.forward(batch_x1, batch_x2, batch_x3)
        mse_train += mlp.backward(batch_x1, batch_x2, batch_x3, batch_y_true)
    mse_train /= (len(x1_train) // batch_size + 1)

    for i in range(0, len(x1_train), 32):
        batch_x1 = x1_train[i:i + 32]
        batch_x2 = x2_train[i:i + 32]
        batch_x3 = x3_train[i:i + 32]
        batch_y_true = y_train[i:i + 32]
        y_pred_batch = mlp_reference.forward(batch_x1, batch_x2, batch_x3)
        mse_fix += mlp_reference.backward(batch_x1, batch_x2, batch_x3, batch_y_true)
    mse_fix /= (len(x1_train) // 32 + 1)

    y_pred_train = mlp.forward(x1_train, x2_train, x3_train)
    y_pred_fix = mlp_reference.forward(x1_train, x2_train, x3_train)
    y_pred_test = mlp.forward(x1_test, x2_test, x3_test)
    y_pred_test_fix = mlp_reference.forward(x1_test, x2_test, x3_test)
    mse_val = np.mean((y_test - y_pred_test) ** 2)

    # Train sklearn model on actual training data
    X_train = np.column_stack((x1_train, x2_train, x3_train))
    X_train_scaled = scaler.transform(to_cpu(X_train))
    X_test = np.column_stack((x1_test, x2_test, x3_test))
    X_test_scaled = scaler.transform(to_cpu(X_test))
    mlp_sklearn.fit(X_train_scaled, to_cpu(y_train))

    # Compute sklearn MSE and validation loss
    y_sklearn_train = mlp_sklearn.predict(X_train_scaled)
    y_sklearn_test = mlp_sklearn.predict(X_test_scaled)
    sklearn_mse = np.mean((to_cpu(y_train) - y_sklearn_train) ** 2)
    sklearn_val_loss = np.mean((to_cpu(y_test) - y_sklearn_test) ** 2)

    # Update histories
    loss_values.append(float(to_cpu(mse_train)))
    loss_values_fix.append(float(to_cpu(mse_fix)))
    val_loss_values.append(float(to_cpu(mse_val)))
    mlp_sklearn_mse.append(float(sklearn_mse))
    mlp_sklearn_val_loss.append(float(sklearn_val_loss))
    epoch_values.append(float(epoch))
    w1_history.append(float(mlp.weights[0][0, 0]))
    w2_history.append(float(mlp.weights[0][0, 1]))
    w3_history.append(float(mlp.weights[0][0, 2]))
    b_history.append(float(mlp.biases[0][0, 0]))

    # Update 3D and contour plots every 5 frames
    if epoch % 5 == 0:
        y_pred_mesh, y_pred_sklearn_mesh = compute_pred_mesh(mlp)
        for coll in ax.collections[2:]:
            coll.remove()
        surface_pred = ax.plot_surface(to_cpu(x1_mesh), to_cpu(x2_mesh), to_cpu(y_pred_mesh), cmap='viridis', alpha=0.3)
        surface_sk = ax.plot_surface(to_cpu(x1_mesh), to_cpu(x2_mesh), y_pred_sklearn_mesh, cmap='cool', alpha=0.3)
        surface_true = ax.plot_surface(to_cpu(x1_mesh), to_cpu(x2_mesh), to_cpu(y_true_mesh), cmap='inferno', alpha=0.3)

        for coll in ax2d.collections:
            coll.remove()
        contour_true = ax2d.contourf(to_cpu(x1_mesh), to_cpu(x2_mesh), to_cpu(y_true_mesh), cmap='inferno', alpha=0.2)
        contour_pred = ax2d.contourf(to_cpu(x1_mesh), to_cpu(x2_mesh), to_cpu(y_pred_mesh), cmap='viridis', alpha=0.2)
        contour_sk = ax2d.contourf(to_cpu(x1_mesh), to_cpu(x2_mesh), y_pred_sklearn_mesh, cmap='cool', alpha=0.2)

    # Update 2D Plot
    choice = 'x1 vs y'  # Simplified for Streamlit; adjust as needed
    ax2d.clear()
    if choice == 'x1 vs y' and w1_true != 0:
        x_plot = np.linspace(np.min(x1_train), np.max(x1_train), 200)
        x2_plot = np.full_like(x_plot, np.mean(x2_train))
        x3_plot = np.full_like(x_plot, np.mean(x3_train))
        y_pred_plot = mlp.forward(x_plot, x2_plot, x3_plot)
        y_pred_fix_plot = mlp_reference.forward(x_plot, x2_plot, x3_plot)
        X_plot = np.column_stack((x_plot, x2_plot, x3_plot))
        X_plot_scaled = scaler.transform(to_cpu(X_plot))
        y_sklearn_plot = mlp_sklearn.predict(X_plot_scaled)
        contour_true = ax2d.contourf(to_cpu(x1_mesh), to_cpu(x2_mesh), to_cpu(y_true_mesh), cmap='inferno', alpha=0.2)
        contour_pred = ax2d.contourf(to_cpu(x1_mesh), to_cpu(x2_mesh), to_cpu(y_pred_mesh), cmap='viridis', alpha=0.2)
        contour_sk = ax2d.contourf(to_cpu(x1_mesh), to_cpu(x2_mesh), y_pred_sklearn_mesh, cmap='cool', alpha=0.2)
        ax2d.scatter(x1_train_cpu, y_train_cpu, label="Training Data", color='blue', marker='o')
        ax2d.plot(to_cpu(x_plot), to_cpu(f_true(x_plot, x2_plot, x3_plot)), label='True', color='hotpink', alpha=0.7)
        ax2d.plot(to_cpu(x_plot), to_cpu(y_pred_plot).ravel(), label='Predicted based on user input', color='lime', alpha=0.7)
        ax2d.plot(to_cpu(x_plot), to_cpu(y_pred_fix_plot).ravel(), label='Predicted based on fixed weights', color='cyan', linestyle='--', alpha=0.7)
        ax2d.plot(to_cpu(x_plot), y_sklearn_plot, label='Sklearn Predicted', color='white', linestyle='--', alpha=0.7)
        ax2d.set_xlabel("x1")
        print(f"Epoch {epoch}: Sklearn y_pred min/max: {y_sklearn_plot.min():.4f}/{y_sklearn_plot.max():.4f}")
    elif choice == 'x2 vs y' and w2_true != 0:
        x_plot = np.linspace(np.min(x2_train), np.max(x2_train), 200)
        x1_plot = np.full_like(x_plot, np.mean(x1_train))
        x3_plot = np.full_like(x_plot, np.mean(x3_train))
        y_pred_plot = mlp.forward(x1_plot, x_plot, x3_plot)
        y_pred_fix_plot = mlp_reference.forward(x1_plot, x_plot, x3_plot)
        X_plot = np.column_stack((x1_plot, x_plot, x3_plot))
        X_plot_scaled = scaler.transform(to_cpu(X_plot))
        y_sklearn_plot = mlp_sklearn.predict(X_plot_scaled)
        ax2d.scatter(x2_train_cpu, y_train_cpu, label="Training Data", color='blue', marker='o')
        ax2d.plot(to_cpu(x_plot), to_cpu(f_true(x1_plot, x_plot, x3_plot)), label='True', color='hotpink', alpha=0.7)
        ax2d.plot(to_cpu(x_plot), to_cpu(y_pred_plot).ravel(), label='Predicted', color='lime', alpha=0.7)
        ax2d.plot(to_cpu(x_plot), y_sklearn_plot, label='Sklearn Predicted', color='black', linestyle='--', alpha=0.7)
        ax2d.plot(to_cpu(x_plot), to_cpu(y_pred_fix_plot).ravel(), label='Predicted based on fixed weights', color='green', linestyle='--', alpha=0.7)
        ax2d.set_xlabel("x2")
        print(f"Epoch {epoch}: Sklearn y_pred min/max: {y_sklearn_plot.min():.4f}/{y_sklearn_plot.max():.4f}")
    elif choice == 'x3 vs y' and w3_true != 0:
        x_plot = np.linspace(np.min(x3_train), np.max(x3_train), 200)
        x1_plot = np.full_like(x_plot, np.mean(x1_train))
        x2_plot = np.full_like(x_plot, np.mean(x2_train))
        y_pred_plot = mlp.forward(x1_plot, x2_plot, x_plot)
        y_pred_fix_plot = mlp_reference.forward(x1_plot, x2_plot, x_plot)
        X_plot = np.column_stack((x1_plot, x2_plot, x_plot))
        X_plot_scaled = scaler.transform(to_cpu(X_plot))
        y_sklearn_plot = mlp_sklearn.predict(X_plot_scaled)
        ax2d.scatter(x3_train_cpu, y_train_cpu, label="Training Data", color='blue', marker='o')
        ax2d.plot(to_cpu(x_plot), to_cpu(f_true(x1_plot, x2_plot, x_plot)), label='True', color='hotpink', alpha=0.7)
        ax2d.plot(to_cpu(x_plot), to_cpu(y_pred_plot).ravel(), label='Predicted', color='lime', alpha=0.7)
        ax2d.plot(to_cpu(x_plot), y_sklearn_plot, label='Sklearn Predicted', color='black', linestyle='--', alpha=0.7)
        ax2d.plot(to_cpu(x_plot), to_cpu(y_pred_fix_plot).ravel(), label='Predicted based on fixed weights', color='green', linestyle='--', alpha=0.7)
        ax2d.set_xlabel("x3")
        print(f"Epoch {epoch}: Sklearn y_pred min/max: {y_sklearn_plot.min():.4f}/{y_sklearn_plot.max():.4f}")
    else:
        print(f"{choice} not valid or variable not in function. Switching to default.")
        return update(epoch)

    ax2d.set_ylabel("y")
    ax2d.grid(True, linestyle="--", alpha=0.5)
    ax2d.legend(loc="upper left")
    ax2d.set_title("2D visualization")

    # Update histogram
    errors = y_train - mlp.forward(x1_train, x2_train, x3_train)
    val_errors = y_test - y_pred_test
    errors_fix = y_train - mlp_reference.forward(x1_train, x2_train, x3_train)
    ax3.clear()
    ax3.hist(to_cpu(errors).ravel(), bins=30, color='red', alpha=0.7, zorder=1, label='Training Errors based on user input')
    ax3.hist(to_cpu(val_errors).ravel(), bins=30, color='lime', alpha=0.7, zorder=3, label='Validation Errors')
    ax3.hist(to_cpu(errors_fix).ravel(), bins=30, color='blue', alpha=0.7, zorder=2, label='Errors with fixed weights')
    ax3.set_xlabel('Error')
    ax3.set_ylabel('Frequency')
    ax3.legend(loc='upper right')

    # Update loss and weights
    loss_line.set_data(epoch_values, loss_values)
    loss_fix_line.set_data(epoch_values, loss_values_fix)
    val_line.set_data(epoch_values, val_loss_values)
    sklearn_mse_line.set_data(epoch_values, mlp_sklearn_mse)
    sklearn_val_line.set_data(epoch_values, mlp_sklearn_val_loss)
    w1_line.set_data(epoch_values, w1_history)
    w2_line.set_data(epoch_values, w2_history)
    w3_line.set_data(epoch_values, w3_history)
    b_line.set_data(epoch_values, b_history)

    # Adjust axes limits
    ax.set_zlim(min(np.min(y_train_cpu), np.min(y_pred_mesh)) - 1, max(np.max(y_train_cpu), np.max(y_pred_mesh)) + 1)
    ax2.set_xlim(0, max(epoch_values) + 1)
    all_losses = loss_values + val_loss_values + mlp_sklearn_mse + mlp_sklearn_val_loss
    ax2.set_ylim(0, max(all_losses) * 1.1 if all_losses else 1)
    ax4.set_ylim(
        min(min(w1_history), min(w2_history), min(w3_history), min(b_history)) - 0.5 if w1_history else -1,
        max(max(w1_history), max(w2_history), max(w3_history), max(b_history)) + 0.5 if w1_history else 1
    )

    # Debugging: Print all metrics
    print(f"Epoch {epoch}: SymbolicMLP mse_train={mse_train:.4f}, mse_val={mse_val:.4f}, "
          f"Sklearn mse_train={sklearn_mse:.4f}, mse_val={sklearn_val_loss:.4f}")
    print(f"Epoch {epoch}: Sklearn MSE={sklearn_mse:.4f}, Val Loss={sklearn_val_loss:.4f}")

    fig.canvas.draw()
# Initialize session state for animation

if 'epoch' not in st.session_state:
    st.session_state.epoch = 0
    st.session_state.max_epoch = 0
    st.session_state.running = False


plot_placeholder = st.empty()


if st.button("Start Training"):
    st.session_state.running = True


if st.button("Stop Training"):
    st.session_state.running = False
    st.session_state.epoch = 0
    st.write("Training stopped!")


if st.session_state.running and st.session_state.epoch < 300:
    with st.spinner("Training in progress..."):
        while st.session_state.epoch < 300 and st.session_state.running:
            update(st.session_state.epoch)
            st.session_state.epoch += 1
            st.session_state.max_epoch = st.session_state.epoch
            fig.canvas.draw()

            # 💡 Plot live aktualisieren
            plot_placeholder.pyplot(fig)

            time.sleep(0.05)

    st.session_state.running = False
    st.write("Training completed!")

if not st.session_state.running:
    plot_placeholder.pyplot(fig)