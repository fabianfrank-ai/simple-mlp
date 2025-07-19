import cupy as cp
import matplotlib.pyplot
import numpy
import matplotlib
import tkinter as tk
from tkinter import ttk,messagebox
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Slider, RadioButtons
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sympy as sp
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D
import threading
from symbolic_backpropagation import symbolicMLP 



if cp.cuda.runtime.getDeviceCount() > 0:
    np=cp
    print("GPU is available, training with cupy")

else:
   np=numpy
   print("Warning! No Cuda-Version found! Will continue training with CPU")
  


# Plot-Setup
fig= plt.figure(figsize=(16,9),)
gs=GridSpec(3,4,height_ratios=[4,2,0.5])
ax=fig.add_subplot(gs[0,0:2],projection='3d')
ax2d=fig.add_subplot(gs[0,2:4])
ax_drop=fig.add_axes([0.88,0.88,0.05,0.1])
ax2=fig.add_subplot(gs[1,0:2])
ax3=fig.add_subplot(gs[1,2:4])
ax.view_init(elev=30, azim=135)


ax.grid(True, linestyle='--', alpha=0.5)
ax2d.grid(True,linestyle='--',alpha=0.5)

ax.set_zlabel('y', fontsize=14)
ax.set_title("Training in Action", fontsize=16)
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.set_title("Loss, Validation Loss and Weights", fontsize=16)
ax3.grid(True, linestyle='--', alpha=0.5)
ax3.set_title("Prediction History", fontsize=16)
ax4 = ax2.twinx()

# Initialize lists for storing history
loss_values = []
val_loss_values = []
epoch_values = []
w1_history = []
w2_history = []
w3_history = []
b_history = []
w4_history=[]
w5_history=[]
w6_history=[]


options=['x1 vs y','x2 vs y', 'x3 vs y']
radio=RadioButtons(ax_drop,options)
selected=[options[0]]

# SymPy symbols and allowed functions
x1_sym, x2_sym, x3_sym = sp.symbols('x1 x2 x3')
w1_sym, w2_sym, w3_sym, w4_sym,w5_sym,w6_sym,w7_sym,w8_sym,w9_sym,w10_sym,w11_sym,w12_sym,w13_sym,w14_sym,w15_sym,w16_sym,b_sym = sp.symbols('w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12 w13 w14 w15 w16 b')
allowed_functions = {
    'sin': sp.sin,
    'cos': sp.cos,
    'exp': sp.exp,
    'log': sp.log,
    'sqrt': sp.sqrt,
    'pi': sp.pi,
}
w_cos={x1_sym:w5_sym,x2_sym:w6_sym,x3_sym:w7_sym}
w_sin={x1_sym:w8_sym,x2_sym:w9_sym,x3_sym:w10_sym}
w_tanh={x1_sym:w11_sym,x2_sym:w12_sym,x3_sym:w13_sym}
w_exp={x1_sym:w14_sym,x2_sym:w15_sym,x3_sym:w16_sym}

def parse_expression(user_input):
    try:
        symbol_dict = {
            str(s): s for s in [
                x1_sym, x2_sym, x3_sym, w1_sym, w2_sym, w3_sym, b_sym
            ]
        }
        symbol_dict.update(allowed_functions)
        
        expr = sp.sympify(user_input, locals=symbol_dict, evaluate=False)

        # nonlinear terms
        nonlinear_terms = {
            sp.sin: {},
            sp.cos: {},
            sp.exp: {},
            sp.tanh: {}
        }

        # search terms and assign weights
        for func, weight_dict in nonlinear_terms.items():
            for term in expr.atoms(func):
                variable = term.args[0]
                if variable in [x1_sym, x2_sym, x3_sym]:
                    weight_symbol = sp.Symbol(f"w_{func.__name__}_{variable}")
                    weight_dict[variable] = weight_symbol

        # extract linear coeficients
        w1_true = expr.diff(x1_sym) if x1_sym in expr.free_symbols else 0
        w2_true = expr.diff(x2_sym) if x2_sym in expr.free_symbols else 0
        w3_true = expr.diff(x3_sym) if x3_sym in expr.free_symbols else 0

        _, b_true = expr.as_independent(x1_sym, x2_sym, x3_sym)

        # Training
        expr_train = w1_sym * x1_sym + w2_sym * x2_sym + w3_sym * x3_sym + b_sym
        for func, weight_dict in nonlinear_terms.items():
            for var, weight in weight_dict.items():
                expr_train += weight * func(var)

        # Lambdify for training and goal
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
        print("Fehler beim Parsen:", e)
        return None





# Get user input
user_input = input("Enter a function in terms of x1, x2, x3 (e.g., 'x1 + x2 + x3 + 2', 'sin(x3) + cos(x2)'): ")
f_train, f_true, w1_true, w2_true, w3_true, b_true,nonlinear_terms ,all_weights= parse_expression(user_input)

if f_train is None:
    print("Invalid input. Using default function: x1 + x2 + x3 + 1")
    user_input = "x1 + x2 + x3 + 1"
    f_train, f_true, w1_true, w2_true, w3_true, b_true,nonlinear_terms, all_weights = parse_expression(user_input)

weight_values = [0] * len(all_weights)

mlp=symbolicMLP(cp,nonlinear_terms)

vis_vars=input("Choose two visualization variables(e.g. x1,x2 or x1,x3)[default:x1,x2]")

if vis_vars not in ['x1,x2','x2,x3','x1,x3']:
    vis_vars='x1,x2'
    print ("Invalid Input. Using default: x1,x2")


# Generate independent random data for x1, x2, x3
np.random.seed(42)
x1 = np.random.uniform(-2, 2, 100)
x2 = np.random.uniform(-2, 2, 100)
x3 = np.random.uniform(-2, 2, 100)
x1_cpu= x1.get()
x2_cpu=x2.get()
x3_cpu=x3.get()


X = np.stack([x1, x2, x3], axis=1)


# Generate true output with noise
y = f_true(x1, x2, x3) + np.random.randn(100) * 0.1  # Adding noise to the true function

# Split data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
x1_train, x2_train, x3_train = x_train[:, 0], x_train[:, 1], x_train[:, 2]
x1_test, x2_test, x3_test = x_test[:, 0], x_test[:, 1], x_test[:, 2]

X_mean= np.mean(x_train,axis=0)
X_std  = np.std(x_train, axis=0) + 1e-8
x1 = (x1 - X_mean[0]) / X_std[0]
x2 = (x2 - X_mean[1]) / X_std[1]
x3 = (x3 - X_mean[2]) / X_std[2]


# Sort for plotting (using x1 as the sorting variable)
sort_idx = np.argsort(x1_train)
x1_sorted = x1_train[sort_idx]
x2_sorted = x2_train[sort_idx]
x3_sorted = x3_train[sort_idx]
y_train_sorted = y_train[sort_idx]
y_true_sorted = f_true(x1_sorted, x2_sorted, x3_sorted)

# Initialize weights
w1, w2, w3,w4,  b = np.random.randn(5) * 0.1  # Smaller initial weights for stability

# Initial prediction
y_pred_sorted = f_train(x1_sorted, x2_sorted, x3_sorted, *weight_values)

# 
x1_plot = np.linspace(np.min(x1_train), np.max(x1_train), 200)  
x2_plot = np.linspace(np.min(x2_train), np.max(x2_train), 200) 
x3_plot =np.linspace(np.min(x3_train), np.max(x3_train), 200) 
y_true_plot = f_true(x1_plot, x2_plot, x3_plot)
y_pred_plot = f_train(x1_plot, x2_plot, x3_plot,*weight_values)


#convert cupy into numpy for matplotlib
x1_train_cpu=x1_train.get()
x2_train_cpu=x2_train.get()
x3_train_cpu=x3_train.get()
y_train_cpu=y_train.get()
x1_test_cpu=x1_test.get()
x2_test_cpu=x2_test.get()
x3_test_cpu=x3_test.get()
y_test_cpu=y_test.get()
x1_plot_cpu=x1_plot.get()
x2_plot_cpu=x2_plot.get()
x3_plot_cpu=x3_plot.get()
y_true_plot_cpu= y_true_plot.get()
y_pred_plot_cpu=y_pred_plot.get()


# Plotting elements
scatter = ax.scatter(x1_train_cpu, x2_train_cpu,y_train_cpu, label="Training Data", color='blue', marker='o')
scatter_2d =ax2d.scatter([],[], label="Training Data", color='blue', marker='o')
scatter_test = ax.scatter(x1_test_cpu,x2_test_cpu, y_test_cpu, label="Test Data", color='cyan', marker='x')
scatter_test_2d = ax2d.scatter([],[],label="Test Data", color='cyan', marker='x')
scatter_pred= ax.scatter(x1_plot_cpu,x2_plot_cpu,y_pred_plot_cpu)
twod_pred=ax2d.plot([],[])
twod_true=ax2d.plot([],[])


if vis_vars== 'x1,x2':

  x1_mesh,x2_mesh= np.meshgrid(np.linspace(np.min(x1_train_cpu),np.max(x1_train_cpu),20),
                              np.linspace(np.min(x2_train_cpu),np.max(x2_train_cpu)))

  x3_mesh = np.full_like(x1_mesh, np.mean(x3_train_cpu))
  ax.set_xlabel('x1', fontsize=14)
  ax.set_ylabel('x2', fontsize=14)
if vis_vars== 'x1,x3':
  

  x1_mesh,x3_mesh= np.meshgrid(np.linspace(np.min(x1_train_cpu),np.max(x1_train_cpu),20),
                              np.linspace(np.min(x2_train_cpu),np.max(x2_train_cpu)))

  x2_mesh=np.full_like(x1_mesh,np.mean(x2_train_cpu))
  ax.set_xlabel('x1', fontsize=14)
  ax.set_ylabel('x3', fontsize=14)
if vis_vars== 'x2,x3':

  x2_mesh,x3_mesh= np.meshgrid(np.linspace(np.min(x1_train_cpu),np.max(x1_train_cpu),20),
                              np.linspace(np.min(x2_train_cpu),np.max(x2_train_cpu)))

  x1_mesh=np.full_like(x2_mesh,np.mean(x1_train_cpu))
  ax.set_xlabel('x2', fontsize=14)
  ax.set_ylabel('x3', fontsize=14)




def compute_pred_mesh():
    global y_pred_mesh
    y_pred_mesh=mlp.forward(x1_mesh,x2_mesh,x3_mesh)

y_pred_mesh=mlp.forward(x1_mesh,x2_mesh,x3_mesh)
y_true_mesh=f_true(x1_mesh,x2_mesh,x3_mesh)


surface_true = ax.plot_surface(x1_mesh.get(), x2_mesh.get(), y_true_mesh.get(), color='hotpink', alpha=0.3, label="True Function")
surface_pred = ax.plot_surface(x1_mesh.get(), x2_mesh.get(), y_pred_mesh.get(), color='lime', alpha=0.3, label="Prediction")
plotted_surface = [None]
surface_first_pred= ax.plot_surface(x1_mesh.get(), x2_mesh.get(), y_pred_mesh.get(), color='orange', alpha=0.3, label="First Prediction")

loss_line, = ax2.plot([], [], color='purple', label='Loss')
val_line, = ax2.plot([], [], color='orange', label='Validation Loss')
w1_line, = ax4.plot([], [], label=f"W1: {w1:.4f}", color='green', linestyle='--', alpha=0.3)
w2_line, = ax4.plot([], [], label=f"W2: {w2:.4f}", color='red', linestyle='--', alpha=0.3)
w3_line, = ax4.plot([], [], label=f"W3: {w3:.4f}", color='blue', linestyle='--', alpha=0.3)
b_line, = ax4.plot([], [], label=f"Bias: {b:.4f}", color='black', linestyle='--', alpha=0.3)

ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')



ax4.set_ylabel('Weights')
ax2.legend(loc='upper right')
ax4.legend(loc='upper left')
ax3.set_xlabel('Error')
ax3.set_ylabel('Frequency')
ax.legend(loc='upper left')

# Slider for learning rate
ax_slider = fig.add_subplot(gs[2,3])
w1_slider = Slider(ax_slider, 'LR', 0.1, 2.0, valinit=0.1)




#callback function
def on_select(label):
   selected[0]=label
   fig.canvas.draw_idle()
   print("Radio button selected:",label)

radio.on_clicked(on_select)


# Update function for animation
def update(epoch):
    global loss_values, val_loss_values, epoch_values, w1_history, w2_history, w3_history, b_history
    lr = w1_slider.val

    # Vorwärtspropagation
    y_pred_train = mlp.forward(x1_train, x2_train, x3_train)  # Shape (80,)
    y_pred_test = mlp.forward(x1_test, x2_test, x3_test)      # Shape (20,)
    

    if np.any(np.isnan(y_pred_train)) or np.any(np.isinf(y_pred_train)) or \
       np.any(np.isnan(y_pred_test)) or np.any(np.isinf(y_pred_test)):
        print("Warning: Predictions contain invalid values. Skipping update.")
        return (scatter, scatter_test, loss_line, val_line, w1_line, w2_line, w3_line, b_line)

    # Loss and gradients
    mse_train = mlp.backward(x1_train, x2_train, x3_train, y_train, lr,y_pred_train)
    mse_val = np.mean((y_test - y_pred_test) ** 2)

    # Save loss and weights
    loss_values.append(float(mse_train))
    val_loss_values.append(float(mse_val))
    epoch_values.append(float(epoch))
    w1_history.append(float(mlp.w_vals['w1']))
    w2_history.append(float(mlp.w_vals['w2']))
    w3_history.append(float(mlp.w_vals['w3']))
    b_history.append(float(mlp.w_vals['b']))

    

    # Update Plots
    loss_line.set_data(epoch_values, loss_values)
    val_line.set_data(epoch_values, val_loss_values)
    w1_line.set_data(epoch_values, w1_history)
    w2_line.set_data(epoch_values, w2_history)
    w3_line.set_data(epoch_values, w3_history)
    b_line.set_data(epoch_values, b_history)
    w1_line.set_label(f"W1: {mlp.w_vals['w1']:.4f}")
    w2_line.set_label(f"W2: {mlp.w_vals['w2']:.4f}")
    w3_line.set_label(f"W3: {mlp.w_vals['w3']:.4f}")
    b_line.set_label(f"Bias: {mlp.w_vals['b']:.4f}")
    loss_line.set_label(f"Loss:{mse_train:.4f}")
    val_line.set_label(f"Val-Loss:{mse_val:.4f}")
    ax4.legend(loc='upper left')

    # Update Mesh
    global surface_pred
    if surface_pred in ax.collections:
        surface_pred.remove()
    y_pred_mesh = mlp.forward(x1_mesh, x2_mesh, x3_mesh)
    surface_pred = ax.plot_surface(x1_mesh.get(), x2_mesh.get(), y_pred_mesh.get(), color='lime', alpha=0.3)

    # Update Histogram
    errors = y_train - y_pred_train
    val_errors = y_test - y_pred_test
    errors_cpu = errors.get() if isinstance(errors, cp.ndarray) else errors
    val_errors_cpu = val_errors.get() if isinstance(val_errors, cp.ndarray) else val_errors
    ax3.clear()
    ax3.hist(errors_cpu, bins=30, color='red', alpha=0.7, label='Training Errors')
    ax3.hist(val_errors_cpu, bins=30, color='lime', alpha=0.7, label='Validation Errors')
    ax3.set_xlabel('Error')
    ax3.set_ylabel('Frequency')
    ax3.legend(loc='upper right')

    # Update 2D Plot
    choice = radio.value_selected
    ax2d.clear()
    if choice == 'x1 vs y':
        if w1_true != 0:
            x_plot = np.linspace(np.min(x1_train), np.max(x1_train), 200)
            x2_plot = np.full_like(x_plot, np.mean(x2_train))
            x3_plot = np.full_like(x_plot, np.mean(x3_train))
            y_pred_plot = mlp.forward(x_plot, x2_plot, x3_plot)
            x_plot_cpu = x_plot.get() if isinstance(x_plot, cp.ndarray) else x_plot
            ax2d.plot(x_plot_cpu, f_true(x_plot, np.mean(x2_train), np.mean(x3_train)).get(), label='True', color='hotpink', alpha=0.7)
            ax2d.scatter(x1_train_cpu, y_train_cpu, label="Training Data", color='blue', marker='o')
            ax2d.plot(x_plot_cpu, y_pred_plot.get(), label='Predicted', color='lime', alpha=0.7)
            ax2d.set_xlabel("x1")
        else:
            print("x1 has not been defined. Switching to another view.")
            radio.set_active(1 if w2_true else 2)
    elif choice == 'x2 vs y':
        if w2_true != 0:
            x_plot = np.linspace(np.min(x2_train), np.max(x2_train), 200)
            x1_plot = np.full_like(x_plot, np.mean(x1_train))
            x3_plot = np.full_like(x_plot, np.mean(x3_train))
            y_pred_plot = mlp.forward(x1_plot, x_plot, x3_plot)
            x_plot_cpu = x_plot.get() if isinstance(x_plot, cp.ndarray) else x_plot
            ax2d.plot(x_plot_cpu, f_true(np.mean(x1_train), x_plot, np.mean(x3_train)).get(), label='True', color='hotpink', alpha=0.7)
            ax2d.scatter(x2_train_cpu, y_train_cpu, label="Training Data", color='blue', marker='o')
            ax2d.plot(x_plot_cpu, y_pred_plot.get(), label='Predicted', color='lime', alpha=0.7)
            ax2d.set_xlabel("x2")
        else:
            print("x2 has not been defined. Switching to another view.")
            radio.set_active(0 if w1_true else 2)
    elif choice == 'x3 vs y':
        if w3_true != 0:
            x_plot = np.linspace(np.min(x3_train), np.max(x3_train), 200)
            x1_plot = np.full_like(x_plot, np.mean(x1_train))
            x2_plot = np.full_like(x_plot, np.mean(x2_train))
            y_pred_plot = mlp.forward(x1_plot, x2_plot, x_plot)
            x_plot_cpu = x_plot.get() if isinstance(x_plot, cp.ndarray) else x_plot
            ax2d.plot(x_plot_cpu, f_true(np.mean(x1_train), np.mean(x2_train), x_plot).get(), label='True', color='hotpink', alpha=0.7)
            ax2d.scatter(x3_train_cpu, y_train_cpu, label="Training Data", color='blue', marker='o')
            ax2d.plot(x_plot_cpu, y_pred_plot.get(), label='Predicted', color='lime', alpha=0.7)
            ax2d.set_xlabel("x3")
        else:
            print("x3 has not been defined. Switching to another view.")
            radio.set_active(0 if w1_true else 1)

    ax2d.set_ylabel("y")
    ax2d.grid(True, linestyle="--", alpha=0.5)
    ax2d.legend(loc="upper left")

    # Adjust axes limits
    ax.set_zlim(min(np.min(y_train.get()), np.min(y_pred_mesh.get())) - 1, max(np.max(y_train.get()), np.max(y_pred_mesh.get())) + 1)
    ax2.set_xlim(0, len(loss_values))
    ylim_max = max(max(loss_values), max(val_loss_values)) * 1.1 if loss_values else 1
    ax2.set_ylim(0, ylim_max)
    ylim_min = min(min(w1_history), min(w2_history), min(w3_history), min(b_history)) - 0.5 if w1_history else -1
    ylim_max_w = max(max(w1_history), max(w2_history), max(w3_history), max(b_history)) + 0.5 if w1_history else 1
    ax4.set_ylim(ylim_min, ylim_max_w)
    print("mse_train:", mse_train, "mse_val:", mse_val)
    print("y_pred_train min/max:", np.min(y_pred_train), np.max(y_pred_train))
    print("y_pred_test min/max:", np.min(y_pred_test), np.max(y_pred_test))   
    return (scatter, scatter_test, loss_line, val_line, w1_line, w2_line, w3_line, b_line)



# Animation
ani = FuncAnimation(fig, update, frames=None, interval=50, blit=False, repeat=False,cache_frame_data=False)
plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
plt.show()