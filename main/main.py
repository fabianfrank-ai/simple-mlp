import cupy as cp
import numpy as np
import matplotlib
import matplotlib.widgets as widgets
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
import sqlite3
from streamlit_autorefresh import st_autorefresh
import time
import sys
import os
import optuna
import seaborn as sns
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from analysis.parameter_analyst import parameter_analysis

def create_database():
    """Create the database table if it doesn't exist."""
    conn = sqlite3.connect('mlp_parameters.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS trainings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            function TEXT,
            optimizer TEXT,
            activation TEXT,
            lr REAL,
            batch_size INTEGER,
            lambda_l2 REAL,
            hidden_layers INTEGER,
            val_loss REAL
        )
    ''')
    conn.commit()
    conn.close()
create_database()

def clear_all_data():
    """Delete all records from the trainings table."""
    conn = sqlite3.connect('mlp_parameters.db')
    c = conn.cursor()
    c.execute("DELETE FROM trainings")
    conn.commit()
    conn.close()
    print("✅ All records deleted from the database.")
#clear_all_data()
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

def get_all_function_names():
    conn = sqlite3.connect("mlp_parameters.db")
    cursor = conn.cursor()
    
    cursor.execute("SELECT function FROM trainings")  
    results = cursor.fetchall()
    
    conn.close()
   
    return [row[0] for row in results]

functions= get_all_function_names()

entries = {func: func for func in get_all_function_names()}



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
            hidden_layers = [st.slider("", 1, 50, 4, step=1, key="hl_slider",help="Specifies the number of neurons or layers between input and output. More hidden units or layers allow the model to learn complex patterns, but also increase the risk of overfitting and training time.")]

with st.sidebar.expander("Solver", expanded=False):
    st.write("Optimization algorithm (Adam, SGD, RMSprop).")
    solver = st.radio("", ('Adam', 'SGD', 'RMSprop'), help="Adam= adaptive moment estimation, SGD= stochastic gradient descent, RMSprop= root mean square propagation.",key="solver_radio")

with st.sidebar.expander("Activation", expanded=False):
    st.write("Activation function (tanh, relu, sigmoid, sin, cos, exp).")
    activation = st.radio("", ('tanh', 'relu', 'sigmoid', 'sin', 'cos', 'exp'), key="activation_radio",help="Non-linearity introduces the ability to learn complex patterns.")
with st.sidebar.expander("Data Base", expanded=True):
        search_query = st.text_input("🔍 Search for function")
        filtered= [entry for entry in entries if search_query in entry.lower()]
        if filtered:
            selected_function = st.selectbox("Select a function", filtered)
            st.success(f"Selected function: {selected_function}")
        else:
            st.warning("No function found matching the query.")
            selected_function = None



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

#dummy nonlinear terms for now, they will be replaced with actual ones later
dummy_nonlinear_terms = {sp.tanh: {x1_sym: sp.Symbol('w_tanh_x1'), x2_sym: sp.Symbol('w_tanh_x2'), x3_sym: sp.Symbol('w_tanh_x3')}}
if 'mlp_reference' not in st.session_state:
  st.session_state.mlp_reference= symbolicMLP(np, dummy_nonlinear_terms, 0.001, 0.01, [3],'adam', 'tanh')
if 'mlp' not in st.session_state:
  st.session_state.mlp = symbolicMLP(np, dummy_nonlinear_terms, 0.001, 0.01, [3],'adam', 'tanh')
if 'model' not in st.session_state:
  st.session_state.model=symbolicMLP(np, dummy_nonlinear_terms, 0.001, 0.01, [3],'adam', 'tanh')
# Update MLP based on selections
if solver == 'Adam':
    st.session_state.mlp.solver = 'adam'
    print("Using Adam optimizer")
elif solver == 'SGD':
    st.session_state.mlp.solver = 'sgd'
    print("Using SGD optimizer")
elif solver == 'RMSprop':
    st.session_state.mlp.solver = 'rmsprop'
    print("Using RMSprop optimizer")

if activation == 'tanh':
    st.session_state.mlp.output_activation = sp.tanh
    print("Using tanh activation")
elif activation == 'relu':
    st.session_state.mlp.output_activation  = 'relu'
    print("Using relu activation")
elif activation == 'sigmoid':
    st.session_state.mlp.output_activation  = 'sigmoid'
    print("Using sigmoid activation")
elif activation == 'sin':
    st.session_state.mlp.output_activation  = sp.sin
    print("Using sin activation")
elif activation == 'cos':
    st.session_state.mlp.output_activation  = sp.cos
    print("Using cos activation")
elif activation == 'exp':
    st.session_state.mlp.output_activation  = sp.exp
    print("Using exp activation")

# User input and visualization variables
if 'user_input' not in st.session_state:
    st.session_state.user_input = "x1 + x2 + x3 + 1"
user_input = st.text_input("Enter a function in terms of x1, x2, x3 (e.g., 'x1 + x2 + x3 + 2', 'sin(x3) + cos(x2)'): ",value=st.session_state.user_input,key="user_input_field")
vis_vars = st.text_input("Choose two visualization variables (e.g., x1,x2 or x1,x3) [default: x1,x2]")

if'batch_size_optuna' not in st.session_state:
    st.session_state.batch_size_optuna = 32



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
        st.session_state.nonlinear_terms = nonlinear_terms  
        st.session_state.all_weights = all_weights
        st.session_state.f_true = sp.lambdify([x1_sym, x2_sym, x3_sym], expr, modules=np)
        st.session_state.f_train = sp.lambdify([x1_sym, x2_sym, x3_sym] + all_weights, expr_train, modules=np)
        st.session_state.w1_true = w1_true
        st.session_state.w2_true = w2_true
        st.session_state.w3_true = w3_true
        st.session_state.b_true = b_true

        return st.session_state.f_train, st.session_state.f_true, w1_true, w2_true, w3_true, b_true, st.session_state.nonlinear_terms, st.session_state.all_weights

    except Exception as e:
        print("Error:", e)
        print("Using default function: x1 + x2 + x3 + 1")
        return parse_expression("x1 + x2 + x3 + 1")

if 'nonlinear_terms' not in st.session_state:
   st.session_state.nonlinear_terms={}
if 'all_weights' not in st.session_state:
   st.session_state.all_weights=[]

weight_values = [0] * len(st.session_state.all_weights)



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

if 'f_true' not in st.session_state:
  st.session_state.f_true = lambda x1, x2, x3: np.zeros_like(x1)
if 'f_train' not in st.session_state:
   st.session_state.f_train = lambda x1, x2, x3: np.zeros_like(x1)
if 'w1_true' not in st.session_state:
    st.session_state.w1_true = 0
if 'w2_true' not in st.session_state:
    st.session_state.w2_true = 0
if 'w3_true' not in st.session_state:
    st.session_state.w3_true = 0
if 'b_true' not in st.session_state:
    st.session_state.b_true = 0
# Generate true output with noise
y = st.session_state.f_true(x1, x2, x3) + np.random.randn(100) * 0.1

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
y_true_sorted = st.session_state.f_true(x1_sorted, x2_sorted, x3_sorted)

# Initial prediction
y_pred_sorted = st.session_state.f_train(x1_sorted, x2_sorted, x3_sorted, *weight_values)

x1_plot = np.linspace(np.min(x1_train), np.max(x1_train), 200)
x2_plot = np.linspace(np.min(x2_train), np.max(x2_train), 200)
x3_plot = np.linspace(np.min(x3_train), np.max(x3_train), 200)
y_true_plot = st.session_state.f_true(x1_plot, x2_plot, x3_plot)
y_pred_plot = st.session_state.f_train(x1_plot, x2_plot, x3_plot, *weight_values)

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

st.session_state.mlp_sklearn = MLPRegressor(hidden_layer_sizes=(200,), max_iter=5000, learning_rate_init=0.001, random_state=42, batch_size=580, warm_start=True)
X_train = np.column_stack((x1_mesh.ravel(), x2_mesh.ravel(), x3_mesh.ravel()))
y_train_mesh = st.session_state.f_train(x1_mesh, x2_mesh, x3_mesh, *weight_values).ravel()
X_train_scaled = scaler.fit_transform(X_train.get())
X_train_np = X_train.get()
y_train_mesh_np = y_train_mesh.get()
X_train_scaled = scaler.fit_transform(X_train_np)
st.session_state.mlp_sklearn.fit(X_train_scaled, y_train_mesh_np)

def compute_pred_mesh(mlp):
    global y_pred_mesh, y_pred_sklearn_mesh
    x1_flat, x2_flat, x3_flat = x1_mesh.ravel(), x2_mesh.ravel(), x3_mesh.ravel()
    y_pred_flat = st.session_state.mlp.forward(x1_flat, x2_flat, x3_flat)
    y_pred_mesh = y_pred_flat.reshape(x1_mesh.shape)
    y_pred_sklearn_flat = st.session_state.mlp_sklearn.predict(X_train.get())
    y_pred_sklearn_mesh = y_pred_sklearn_flat.reshape(x1_mesh.shape)
    return y_pred_mesh, y_pred_sklearn_mesh

y_pred_mesh, y_pred_sklearn_mesh = compute_pred_mesh(st.session_state.mlp)
y_true_mesh = st.session_state.f_true(x1_mesh, x2_mesh, x3_mesh)

ax.plot([], [], [], color='white', alpha=0.3, label='Sklearn Prediction')
surface_first_pred = ax.plot_surface(x1_mesh.get(), x2_mesh.get(), y_pred_mesh.get(), cmap='viridis', alpha=0.3, label="First Prediction")
loss_line, = ax2.plot([], [], color='purple', label='Loss')
loss_fix_line, = ax2.plot([], [], color='palegreen', label='Loss with fixed parameters')
val_line, = ax2.plot([], [], color='orange', label='Validation Loss')
sklearn_line = ax2d.plot([], [], color='white', label="sklearn prediction", linestyle='--')
sklearn_mse_line, = ax2.plot([], [], color='black', label='Sklearn MSE')
sklearn_val_line, = ax2.plot([], [], color='gray', label='Sklearn Val Loss')
w1_line, = ax4.plot([], [], label=f"W1: {st.session_state.mlp.weights[0][0, 0]:.4f}", color='green', linestyle='--', alpha=0.3)
w2_line, = ax4.plot([], [], label=f"W2: {st.session_state.mlp.weights[0][0, 1]:.4f}", color='red', linestyle='--', alpha=0.3)
w3_line, = ax4.plot([], [], label=f"W3: {st.session_state.mlp.weights[0][0, 2]:.4f}", color='blue', linestyle='--', alpha=0.3)
b_line, = ax4.plot([], [], label=f"Bias: {st.session_state.mlp.biases[0][0, 0]:.4f}", color='black', linestyle='--', alpha=0.3)
optuna_line, = ax4.plot([], [], label='Optuna Loss', color= '#FFFF00', linestyle='--')

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

# Initialize line plots with empty data to prevent broadcast errors
loss_line.set_data([0], [0])
loss_fix_line.set_data([0], [0])
val_line.set_data([0], [0])
sklearn_mse_line.set_data([0], [0])
sklearn_val_line.set_data([0], [0])
w1_line.set_data([0], [0])
w2_line.set_data([0], [0])
w3_line.set_data([0], [0])
b_line.set_data([0], [0])
optuna_line.set_data([0], [0])

def to_cpu(arr):
    return arr.get() if isinstance(arr, cp.ndarray) else arr


# Update function for animation
plotted_surfaces = {'pred': None, 'sk': None, 'true': None}

# Initialize lists for storing history
if 'loss_values' not in st.session_state:
 st.session_state.loss_values = []
if 'loss_values_fix' not in st.session_state:
 st.session_state.loss_values_fix = []
if 'val_loss_values' not in st.session_state:
 st.session_state.val_loss_values = []
if 'epoch_values' not in st.session_state:
 st.session_state.epoch_values = []
if 'w1_history' not in st.session_state:
 st.session_state.w1_history = []
if 'w2_history' not in st.session_state:
 st.session_state.w2_history = []
if 'w3_history' not in st.session_state:
 st.session_state.w3_history = []
if 'b_history' not in st.session_state:
 st.session_state.b_history = []
if 'mlp_sklearn_mse' not in st.session_state:
 st.session_state.mlp_sklearn_mse = []
if 'mlp_sklearn_val_loss' not in st.session_state:
 st.session_state.mlp_sklearn_val_loss = []
if 'w_history' not in st.session_state:
 st.session_state.w_history=[]
if 'b_all_history' not in st.session_state:
 st.session_state.b_all_history=[]
if 'mse_optuna' not in st.session_state:
    st.session_state.mse_optuna = []
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = None
if 'analyzer' in st.session_state:
 st.session_state.analyzer= parameter_analysis(st.session_state.w_history,st.session_state.b_all_history,lr, lambda_l2, hidden_layers, 300, batch_size, solver, activation, st.session_state.loss_values, st.session_state.val_loss_values)

def objective(trial):
    optimizer_name = trial.suggest_categorical('optimizer', ['sgd', 'adam', 'rmsprop'])
    activation = trial.suggest_categorical('activation', ['relu', 'tanh','sigmoid','sin', 'cos', 'exp', 'log'])
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    batch_size_optuna = trial.suggest_int('batch_size', 16, 128)
    lambda_l2 = trial.suggest_loguniform('lambda_l2', 1e-8, 1e-1)
    hidden_layers = trial.suggest_int('hidden_layers', 1, 10)
    
    st.session_state.model=symbolicMLP(np,dummy_nonlinear_terms,lambda_l2,lr,hidden_layers, solver=optimizer_name, activation_function=activation)
    val_loss_optuna=train_and_evaluate(st.session_state.model,batch_size_optuna,trial)
    return val_loss_optuna

def train_and_evaluate(model, batch_size,trial):
    epochs = 100
    for epoch in range(epochs):
        mse_train = 0.0
        # Training in Batches
        for i in range(0, len(x1_train), batch_size):
            batch_x1 = x1_train[i:i + batch_size]
            batch_x2 = x2_train[i:i + batch_size]
            batch_x3 = x3_train[i:i + batch_size]
            batch_y_true = y_train[i:i + batch_size]
            y_pred_batch = model.forward(batch_x1, batch_x2, batch_x3)
            mse_train += model.backward(batch_x1, batch_x2, batch_x3, batch_y_true)
        mse_train /= (len(x1_train) // batch_size + 1)

        y_pred_val = model.forward(x1_test, x2_test, x3_test)
        mse_val = np.mean((y_test - y_pred_val) ** 2)
        trial.report(mse_val, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return mse_val
def function_exists(function_str):
    conn = sqlite3.connect('mlp_parameters.db')
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM trainings WHERE function = ?", (function_str,))
    exists = c.fetchone()[0] > 0
    conn.close()
    return exists 

def get_existing_entry(function_str):
    conn = sqlite3.connect('mlp_parameters.db')
    c = conn.cursor()
    c.execute("SELECT * FROM trainings WHERE function = ?", (function_str,))
    result = c.fetchone()
    conn.close()
    return result

def insert_data(function, best_params, val_loss):
    conn = sqlite3.connect('mlp_parameters.db')
    c = conn.cursor()

    c.execute('''
        SELECT * FROM trainings
        WHERE function = ? AND optimizer = ? AND activation = ? AND
              lr = ? AND batch_size = ? AND lambda_l2 = ? AND hidden_layers = ? AND val_loss = ?
    ''', (
        function,
        best_params['optimizer'],
        best_params['activation'],
        best_params['lr'],
        best_params['batch_size'],
        best_params['lambda_l2'],
        best_params['hidden_layers'],
        val_loss
    ))

    if c.fetchone() is None:
        c.execute('''
            INSERT INTO trainings (
                function, optimizer, activation, lr,
                batch_size, lambda_l2, hidden_layers, val_loss
            ) VALUES (?, ?, ?, ?, ?, ?, ?,?)
        ''', (
            function,
            best_params['optimizer'],
            best_params['activation'],
            best_params['lr'],
            best_params['batch_size'],
            best_params['lambda_l2'],
            best_params['hidden_layers'],
            val_loss
        ))
        conn.commit()
        print("✅ New combination inserted.")
    else:
        print("⚠️ Combination already exists. Skipping insert.")

    conn.close()

consent=st.checkbox("I consent, that the data used in this process will be used for optimization and evaluation by agreeing for my data to be saved in the database.")

if consent:
    st.success("Consent given! Thank you for your contribution to this project!")
else: 
    st.warning("Consent not given! Your data will not be saved.Please rerun if you change your mind.")




def plot_weight_heatmap(weight_matrix,layer_index):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(weight_matrix, cmap="coolwarm",center=0,annot=False,xticklabels=True, yticklabels=True,cbar=True, ax=ax)
    ax.set_title(f"Weight Heatmap for Layer {layer_index+1}")
    ax.set_xlabel("Inputs")
    ax.set_ylabel("Neurons")



#st.markdown("**Input->Hidden Layers**")
#fig1= plot_weight_heatmap(st.session_state.mlp_reference.weights[0].get(), 0)
#st.pyplot(fig1)
#st.markdown("**Hidden Layers->Output Layer**")
#fig2= plot_weight_heatmap(st.session_state.mlp_reference.weights[-1].get(), 1)
#st.pyplot(fig2)

def update(epoch):
    global loss_values, val_loss_values, epoch_values, w1_history, w2_history, w3_history, b_history, batch_size_optuna
    global mlp_sklearn_val_loss, mlp_sklearn_mse
    global w1_true, w2_true, w3_true, b_true 
    lr = st.session_state.lr_slider
    batch_size = st.session_state.batch_slider
    x1_flat, x2_flat, x3_flat = x1_mesh.ravel(), x2_mesh.ravel(), x3_mesh.ravel()
    y_true_flat = st.session_state.f_true(x1_flat, x2_flat, x3_flat)
    mse_train = 0.0
    mse_fix = 0.0
    mse_optuna = 0.0
    y_pred_mesh, y_pred_sklearn_mesh = compute_pred_mesh(st.session_state.mlp)
    # Train symbolicMLP in batches
    for i in range(0, len(x1_train), batch_size):
        batch_x1 = x1_train[i:i + batch_size]
        batch_x2 = x2_train[i:i + batch_size]
        batch_x3 = x3_train[i:i + batch_size]
        batch_y_true = y_train[i:i + batch_size]
        y_pred_batch = st.session_state.mlp.forward(batch_x1, batch_x2, batch_x3)
        mse_train += st.session_state.mlp.backward(batch_x1, batch_x2, batch_x3, batch_y_true)
    mse_train /= (len(x1_train) // batch_size + 1)

    for i in range(0, len(x1_train), 32):
        batch_x1 = x1_train[i:i + 32]
        batch_x2 = x2_train[i:i + 32]
        batch_x3 = x3_train[i:i + 32]
        batch_y_true = y_train[i:i + 32]
        y_pred_batch = st.session_state.mlp_reference.forward(batch_x1, batch_x2, batch_x3)
        mse_fix += st.session_state.mlp_reference.backward(batch_x1, batch_x2, batch_x3, batch_y_true)
    mse_fix /= (len(x1_train) // 32 + 1)

    for i in range(0, len(x1_train),st.session_state.batch_size_optuna):
        batch_x1 = x1_train[i:i + st.session_state.batch_size_optuna]
        batch_x2 = x2_train[i:i + st.session_state.batch_size_optuna]
        batch_x3 = x3_train[i:i + st.session_state.batch_size_optuna]
        batch_y_true = y_train[i:i + st.session_state.batch_size_optuna]
        y_pred_batch = st.session_state.model.forward(batch_x1, batch_x2, batch_x3)
        mse_optuna += st.session_state.model.backward(batch_x1, batch_x2, batch_x3, batch_y_true)
    mse_optuna /= (len(x1_train) // st.session_state.batch_size_optuna + 1)


    y_pred_train =st.session_state.mlp.forward(x1_train, x2_train, x3_train)
    y_pred_fix = st.session_state.mlp_reference.forward(x1_train, x2_train, x3_train)
    y_pred_test = st.session_state.mlp.forward(x1_test, x2_test, x3_test)
    y_pred_test_fix = st.session_state.mlp_reference.forward(x1_test, x2_test, x3_test)
    mse_val = np.mean((y_test - y_pred_test) ** 2)

    # Train sklearn model on actual training data
    X_train = np.column_stack((x1_train, x2_train, x3_train))
    X_train_scaled = scaler.transform(to_cpu(X_train))
    X_test = np.column_stack((x1_test, x2_test, x3_test))
    X_test_scaled = scaler.transform(to_cpu(X_test))
    st.session_state.mlp_sklearn.fit(X_train_scaled, to_cpu(y_train))

    # Compute sklearn MSE and validation loss
    y_sklearn_train = st.session_state.mlp_sklearn.predict(X_train_scaled)
    y_sklearn_test = st.session_state.mlp_sklearn.predict(X_test_scaled)
    sklearn_mse = np.mean((to_cpu(y_train) - y_sklearn_train) ** 2)
    sklearn_val_loss = np.mean((to_cpu(y_test) - y_sklearn_test) ** 2)

    # Update histories
    st.session_state.loss_values.append(float(to_cpu(mse_train)))
    st.session_state.loss_values_fix.append(float(to_cpu(mse_fix)))
    st.session_state.val_loss_values.append(float(to_cpu(mse_val)))
    st.session_state.mlp_sklearn_mse.append(float(sklearn_mse))
    st.session_state.mlp_sklearn_val_loss.append(float(sklearn_val_loss))
    st.session_state.epoch_values.append(float(epoch))
    st.session_state.w1_history.append(float(st.session_state.mlp.weights[0][0, 0]))
    st.session_state.w2_history.append(float(st.session_state.mlp.weights[0][0, 1]))
    st.session_state.w3_history.append(float(st.session_state.mlp.weights[0][0, 2]))
    st.session_state.b_history.append(float(st.session_state.mlp.biases[0][0, 0]))
    st.session_state.b_all_history.append(st.session_state.mlp.biases.copy())
    st.session_state.mse_optuna.append(float(mse_optuna))
    st.session_state.w_history.append(st.session_state.mlp.weights.copy())




    # Update 3D and contour plots every 5 frames
    if epoch % 5 == 0:
        y_pred_mesh, y_pred_sklearn_mesh = compute_pred_mesh(st.session_state.mlp)
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
    if choice == 'x1 vs y' and st.session_state.w1_true != 0:
        x_plot = np.linspace(np.min(x1_train), np.max(x1_train), 200)
        x2_plot = np.full_like(x_plot, np.mean(x2_train))
        x3_plot = np.full_like(x_plot, np.mean(x3_train))
        y_pred_plot = st.session_state.mlp.forward(x_plot, x2_plot, x3_plot)
        y_pred_fix_plot = st.session_state.mlp_reference.forward(x_plot, x2_plot, x3_plot)
        y_ideal_plot=st.session_state.model.forward(x_plot, x2_plot, x3_plot)
        X_plot = np.column_stack((x_plot, x2_plot, x3_plot))
        X_plot_scaled = scaler.transform(to_cpu(X_plot))
        y_sklearn_plot = st.session_state.mlp_sklearn.predict(X_plot_scaled)
        contour_true = ax2d.contourf(to_cpu(x1_mesh), to_cpu(x2_mesh), to_cpu(y_true_mesh), cmap='inferno', alpha=0.2)
        contour_pred = ax2d.contourf(to_cpu(x1_mesh), to_cpu(x2_mesh), to_cpu(y_pred_mesh), cmap='viridis', alpha=0.2)
        contour_sk = ax2d.contourf(to_cpu(x1_mesh), to_cpu(x2_mesh), y_pred_sklearn_mesh, cmap='cool', alpha=0.2)
        ax2d.scatter(x1_train_cpu, y_train_cpu, label="Training Data", color='blue', marker='o')
        ax2d.plot(to_cpu(x_plot), to_cpu(st.session_state.f_true(x_plot, x2_plot, x3_plot)), label='True', color='hotpink', alpha=0.7)
        ax2d.plot(to_cpu(x_plot), to_cpu(y_pred_plot).ravel(), label='Predicted based on user input', color='lime', alpha=0.7)
        ax2d.plot(to_cpu(x_plot), to_cpu(y_ideal_plot).ravel(), label='Prediction with ideal parameters', color='#FFA500', linestyle='--', alpha=0.7)
        ax2d.plot(to_cpu(x_plot), to_cpu(y_pred_fix_plot).ravel(), label='Predicted based on fixed weights', color='cyan', linestyle='--', alpha=0.7)
        ax2d.plot(to_cpu(x_plot), y_sklearn_plot, label='Sklearn Predicted', color='white', linestyle='--', alpha=0.7)
        ax2d.set_xlabel("x1")
        print(f"Epoch {epoch}: Sklearn y_pred min/max: {y_sklearn_plot.min():.4f}/{y_sklearn_plot.max():.4f}")
    elif choice == 'x2 vs y' and st.session_state.w2_true != 0:
        x_plot = np.linspace(np.min(x2_train), np.max(x2_train), 200)
        x1_plot = np.full_like(x_plot, np.mean(x1_train))
        x3_plot = np.full_like(x_plot, np.mean(x3_train))
        y_pred_plot = st.session_state.mlp.forward(x1_plot, x_plot, x3_plot)
        y_pred_fix_plot = st.session_state.mlp_reference.forward(x1_plot, x_plot, x3_plot)
        X_plot = np.column_stack((x1_plot, x_plot, x3_plot))
        X_plot_scaled = scaler.transform(to_cpu(X_plot))
        y_sklearn_plot = st.session_state.mlp_sklearn.predict(X_plot_scaled)
        ax2d.scatter(x2_train_cpu, y_train_cpu, label="Training Data", color='blue', marker='o')
        ax2d.plot(to_cpu(x_plot), to_cpu(st.session_state.f_true(x1_plot, x_plot, x3_plot)), label='True', color='hotpink', alpha=0.7)
        ax2d.plot(to_cpu(x_plot), to_cpu(y_pred_plot).ravel(), label='Predicted', color='lime', alpha=0.7)
        ax2d.plot(to_cpu(x_plot), y_sklearn_plot, label='Sklearn Predicted', color='black', linestyle='--', alpha=0.7)
        ax2d.plot(to_cpu(x_plot), to_cpu(y_pred_fix_plot).ravel(), label='Predicted based on fixed weights', color='green', linestyle='--', alpha=0.7)
        ax2d.set_xlabel("x2")
        print(f"Epoch {epoch}: Sklearn y_pred min/max: {y_sklearn_plot.min():.4f}/{y_sklearn_plot.max():.4f}")
    elif choice == 'x3 vs y' and st.session_state.w3_true != 0:
        x_plot = np.linspace(np.min(x3_train), np.max(x3_train), 200)
        x1_plot = np.full_like(x_plot, np.mean(x1_train))
        x2_plot = np.full_like(x_plot, np.mean(x2_train))
        y_pred_plot = st.session_state.mlp.forward(x1_plot, x2_plot, x_plot)
        y_pred_fix_plot = st.session_state.mlp_reference.forward(x1_plot, x2_plot, x_plot)
        X_plot = np.column_stack((x1_plot, x2_plot, x_plot))
        X_plot_scaled = scaler.transform(to_cpu(X_plot))
        y_sklearn_plot = st.session_state.mlp_sklearn.predict(X_plot_scaled)
        ax2d.scatter(x3_train_cpu, y_train_cpu, label="Training Data", color='blue', marker='o')
        ax2d.plot(to_cpu(x_plot), to_cpu(st.session_state.f_true(x1_plot, x2_plot, x_plot)), label='True', color='hotpink', alpha=0.7)
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
    errors = y_train - st.session_state.mlp.forward(x1_train, x2_train, x3_train)
    val_errors = y_test - y_pred_test
    errors_fix = y_train - st.session_state.mlp_reference.forward(x1_train, x2_train, x3_train)
    ax3.clear()
    ax3.hist(to_cpu(errors).ravel(), bins=30, color='red', alpha=0.7, zorder=1, label='Training Errors based on user input')
    ax3.hist(to_cpu(val_errors).ravel(), bins=30, color='lime', alpha=0.7, zorder=3, label='Validation Errors')
    ax3.hist(to_cpu(errors_fix).ravel(), bins=30, color='blue', alpha=0.7, zorder=2, label='Errors with fixed weights')
    ax3.set_xlabel('Error')
    ax3.set_ylabel('Frequency')
    ax3.legend(loc='upper right')

    # Update loss and weights
    loss_line.set_data(st.session_state.epoch_values, st.session_state.loss_values)
    loss_fix_line.set_data(st.session_state.epoch_values, st.session_state.loss_values_fix)
    val_line.set_data(st.session_state.epoch_values, st.session_state.val_loss_values)
    sklearn_mse_line.set_data(st.session_state.epoch_values, st.session_state.mlp_sklearn_mse)
    sklearn_val_line.set_data(st.session_state.epoch_values, st.session_state.mlp_sklearn_val_loss)
    w1_line.set_data(st.session_state.epoch_values, st.session_state.w1_history)
    w2_line.set_data(st.session_state.epoch_values, st.session_state.w2_history)
    w3_line.set_data(st.session_state.epoch_values, st.session_state.w3_history)
    b_line.set_data(st.session_state.epoch_values, st.session_state.b_history)
    optuna_line.set_data(st.session_state.epoch_values, st.session_state.mse_optuna)

    # Adjust axes limits
    ax.set_zlim(min(np.min(y_train_cpu), np.min(y_pred_mesh)) - 1, max(np.max(y_train_cpu), np.max(y_pred_mesh)) + 1)
    ax2.set_xlim(0, max(st.session_state.epoch_values) + 1)
    all_losses = st.session_state.loss_values + st.session_state.val_loss_values + st.session_state.mlp_sklearn_mse + st.session_state.mlp_sklearn_val_loss
    ax2.set_ylim(0, max(all_losses) * 1.1 if all_losses else 1)
    ax4.set_ylim(
        min(min(st.session_state.w1_history), min(st.session_state.w2_history), min(st.session_state.w3_history), min(st.session_state.b_history)) - 0.5 if st.session_state.w1_history else -1,
        max(max(st.session_state.w1_history), max(st.session_state.w2_history), max(st.session_state.w3_history), max(st.session_state.b_history)) + 0.5 if st.session_state.w1_history else 1
    )

    # Debugging: Print all metrics
    print(f"Epoch {epoch}: SymbolicMLP mse_train={mse_train:.4f}, mse_val={mse_val:.4f}, "
          f"Sklearn mse_train={sklearn_mse:.4f}, mse_val={sklearn_val_loss:.4f}")
    print(f"Epoch {epoch}: Sklearn MSE={sklearn_mse:.4f}, Val Loss={sklearn_val_loss:.4f}")

    fig.canvas.draw()



# Initialize session state for animation
if 'epoch' not in st.session_state:
    st.session_state.epoch = 0
if 'max_epoch' not in st.session_state:
    st.session_state.max_epoch = 0
if 'running' not in st.session_state:
    st.session_state.running = False
if 'initialized' not in st.session_state:
    st.session_state.initialized = False  # Ensure initialized is always set
if 'optimization_done' not in st.session_state:
    st.session_state.optimization_done = False

# Placeholder for dynamic plot update
plot_placeholder = st.empty()

if selected_function is not None:
  st.table(get_existing_entry(selected_function))

# Start Training button
if st.button("Start Training"):
    if not st.session_state.running:  # Only start if not already running
        st.session_state.running = True
        st.session_state.epoch = st.session_state.max_epoch
        st.session_state.initialized = False  # Reset initialization flag
if user_input != st.session_state.user_input:
    st.session_state.user_input = user_input
    st.session_state.running = False
    st.session_state.epoch = 0
    st.session_state.max_epoch = 0
    st.session_state.initialized = False
    st.session_state.optimization_done = False
    # Clear existing histories
    st.session_state.loss_values = []
    st.session_state.val_loss_values = []
    st.session_state.epoch_values = []
    st.session_state.w1_history = []
    st.session_state.w2_history = []
    st.session_state.w3_history = []
    st.session_state.b_history = []
    st.session_state.mse_optuna=[]

col1, col2, col3 = st.columns([1, 2, 1])
plot_placeholder=col2.empty()

# Stop Training button
if st.button("Stop Training"):
    st.session_state.optimization_done = False
    st.session_state.running = False
    st.session_state.epoch = 0  # Reset for next run
    st.session_state.max_epoch = 0
    st.session_state.initialized = False

# Hyperparameter optimization (run only at start)
# Initialization (run only once at start)
if st.session_state.running and not st.session_state.initialized:
    f_train, f_true, w1_true, w2_true, w3_true, b_true, nonlinear_terms, all_weights = parse_expression(st.session_state.user_input)
    
    # Initialize MLPs with the parsed nonlinear terms
    st.session_state.mlp = symbolicMLP(np, st.session_state.nonlinear_terms, lambda_l2, lr, hidden_layers, solver, activation)
    st.session_state.mlp_reference = symbolicMLP(np, st.session_state.nonlinear_terms, 0.001, 0.01, [50], 'adam', 'placeholder')
    
    # Handle existing function case
    if function_exists(user_input):
        entry = get_existing_entry(st.session_state.user_input)
        st.session_state.model = symbolicMLP(np, st.session_state.nonlinear_terms, entry[6], entry[4], entry[7], entry[2], entry[3])
        st.session_state.batch_size_optuna = entry[5]
        st.session_state.optimization_done = True  # Set optimization as done for existing functions
    
    st.session_state.initialized = True

# Hyperparameter optimization (run only at start)
if st.session_state.running and not st.session_state.optimization_done and consent:
    if not function_exists(st.session_state.user_input):
        with st.spinner("Optimizing hyperparameters..."):
            study = optuna.create_study(direction="minimize",sampler=optuna.samplers.CmaEsSampler(warn_independent_sampling=False), pruner=optuna.pruners.MedianPruner(n_warmup_steps=10))
            for _ in range(20):  # Retry optimization 20 times in case of failures
                try:
                    study.optimize(objective, n_trials=100, n_jobs=8,)
                    break  # Exit loop after successful optimization
                except optuna.exceptions.TrialPruned:
                    continue

            if study.best_trial is not None:
                best_params = study.best_params
                best_val_loss = study.best_value
                insert_data(user_input, best_params, best_val_loss)
                st.session_state.batch_size_optuna = best_params['batch_size']
                st.session_state.model = symbolicMLP(np, st.session_state.nonlinear_terms, best_params['lambda_l2'], best_params['lr'], best_params['hidden_layers'], best_params['optimizer'], best_params['activation'])
                st.session_state.optimization_done = True
            else:
                st.warning("No trial found")
                st.session_state.optimization_done = True  # Prevent infinite loop
    else:
        entry = get_existing_entry(st.session_state.user_input)
        st.session_state.model = symbolicMLP(np, st.session_state.nonlinear_terms, entry[6], entry[4], entry[7], entry[2], entry[3])
        st.session_state.batch_size_optuna = entry[5]
        st.write("Function already exists. Skipping hyperparameter optimization.")
        st.session_state.optimization_done = True


# Animation loop (run incrementally) - Add safety check for empty arrays
if (st.session_state.running and 
    st.session_state.optimization_done and 
    st.session_state.initialized and 
    st.session_state.epoch < 300):
    
    # Only proceed if we have the necessary data
    if (hasattr(st.session_state, 'mlp') and 
        hasattr(st.session_state, 'model') and 
        len(st.session_state.epoch_values) >= 0):
        
        update(st.session_state.epoch)  # Update the figure
        st.session_state.epoch += 1
        st.session_state.max_epoch = st.session_state.epoch

        st.pyplot(fig)  # Update the plot in-place
       

        time.sleep(0.05)  # Control animation speed
        st.rerun()  # Add this line to continue the animatio
   

# Completion check
if st.session_state.running and st.session_state.epoch == 300:
    st.session_state.running = False
    st.write("Training completed!")
    report = st.session_state.analyzer.generate_training_report()
    st.markdown("### 🧠 Analysis of your training")
    st.code(report, language="text")

# Display initial or paused plot
if not st.session_state.running:
 
     plot_placeholder.pyplot(fig)

  