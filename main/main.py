import cupy as cp
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

mlp=symbolicMLP(np_module=cp)


if cp.cuda.runtime.getDeviceCount() > 0:
    np=cp
    print("GPU is available, training with cupy")

else:
   np=numpy
   print("Warning! No Cuda-Version found! Will continue training with CPU")
  


# Plot-Setup
fig= plt.figure(figsize=(16,14),)
gs=GridSpec(4,2,height_ratios=[3,1,1,1])
ax=fig.add_subplot(gs[0:3,0],projection='3d')
ax2d=fig.add_subplot(gs[0:3,1])
ax_drop=fig.add_axes([0.88,0.88,0.05,0.1])
ax2=fig.add_subplot(gs[3,0])
ax3=fig.add_subplot(gs[3,1])
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
w1_sym, w2_sym, w3_sym, w4_sym,w5_sym,w6_sym,b_sym = sp.symbols('w1 w2 w3 w4 w5 w6 b')
allowed_functions = {
    'sin': sp.sin,
    'cos': sp.cos,
    'exp': sp.exp,
    'log': sp.log,
    'sqrt': sp.sqrt,
    'pi': sp.pi,
}

def parse_expression(user_input):
    try:
        symbol_dict = {str(s): s for s in [x1_sym, x2_sym, x3_sym, w1_sym, w2_sym, w3_sym, b_sym]}
        symbol_dict.update(allowed_functions)
        expr = sp.sympify(user_input, locals=symbol_dict, evaluate=False)
        # Extract coefficients and bias
        w1_true = expr.coeff(x1_sym) if x1_sym in expr.free_symbols else 0
        w2_true = expr.coeff(x2_sym) if x2_sym in expr.free_symbols else 0
        w3_true = expr.coeff(x3_sym) if x3_sym in expr.free_symbols else 0



        _, b_true = expr.as_independent(x1_sym, x2_sym, x3_sym)
        
        # Create training expression
        expr_train =w1_sym * x1_sym + w2_sym * x2_sym + w3_sym * x3_sym + b_sym
        terms = {
                 sp.cos(x3_sym): w4_sym * sp.cos(x3_sym),
                 sp.sin(x2_sym): w5_sym * sp.sin(x2_sym),
                 x1_sym**2: w6_sym * x1_sym**2
}
        for term, coef_term in terms.items():
           if expr.has(term):
            expr_train += coef_term
        
        
        f_true = sp.lambdify([x1_sym, x2_sym, x3_sym], expr, modules=np)
        f_train = sp.lambdify([x1_sym, x2_sym, x3_sym, w1_sym, w2_sym, w3_sym,w4_sym,w5_sym,w6_sym, b_sym], expr_train, modules=np)
        
        return f_train, f_true, w1_true, w2_true, w3_true, b_true
    except Exception as e:
        print(f"Error parsing expression: {e}")
        return None, None, None, None, None, None

# Get user input
user_input = input("Enter a function in terms of x1, x2, x3 (e.g., 'x1 + x2 + x3 + 2', 'sin(x3) + cos(x2)'): ")
f_train, f_true, w1_true, w2_true, w3_true, b_true = parse_expression(user_input)

if f_train is None:
    print("Invalid input. Using default function: x1 + x2 + x3 + 1")
    user_input = "x1 + x2 + x3 + 1"
    f_train, f_true, w1_true, w2_true, w3_true, b_true = parse_expression(user_input)


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


# Sort for plotting (using x1 as the sorting variable)
sort_idx = np.argsort(x1_train)
x1_sorted = x1_train[sort_idx]
x2_sorted = x2_train[sort_idx]
x3_sorted = x3_train[sort_idx]
y_train_sorted = y_train[sort_idx]
y_true_sorted = f_true(x1_sorted, x2_sorted, x3_sorted)

# Initialize weights
w1, w2, w3,w4,w6,w5, b = np.random.randn(7) * 0.1  # Smaller initial weights for stability

# Initial prediction
y_pred_sorted = f_train(x1_sorted, x2_sorted, x3_sorted, w1, w2, w3,w4,w5,w6, b)

# 
x1_plot = np.linspace(np.min(x1_train), np.max(x1_train), 200)  
x2_plot = np.linspace(np.min(x2_train), np.max(x2_train), 200) 
x3_plot =np.linspace(np.min(x3_train), np.max(x3_train), 200) 
y_true_plot = f_true(x1_plot, x2_plot, x3_plot)
y_pred_plot = f_train(x1_plot, x2_plot, x3_plot, w1, w2, w3, w4,w5,w6,b)


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
    y_pred_mesh=f_train(x1_mesh,x2_mesh,x3_mesh,w1,w2,w3,w4,w5,w6,b)

y_pred_mesh=f_train(x1_mesh,x2_mesh,x3_mesh,w1,w2,w3,w4,w5,w6,b)
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
w4_line, = ax4.plot([], [], label=f"W4: {w4:.4f}", color='cyan', linestyle='--', alpha=0.3)
w5_line, = ax4.plot([], [], label=f"W5: {w5:.4f}", color='hotpink', linestyle='--', alpha=0.3)
w6_line, = ax4.plot([], [], label=f"W6: {w6:.4f}", color='yellow', linestyle='--', alpha=0.3)
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
ax_slider = fig.add_axes([0.15, 0.05, 0.3, 0.03])
w1_slider = Slider(ax_slider, 'LR', 0.0001, 0.01, valinit=0.001)

#Buttons


#callback function
def on_select(label):
   selected[0]=label
   fig.canvas.draw_idle()
   print("Radio button selected:",label)

radio.on_clicked(on_select)


# Update function for animation
def update(epoch):
    global w1, w2, w3,w4,w5,w6, b
    lr = w1_slider.val
    decay_factor = 0.5

    # Compute predictions and errors
    y_pred_train = mlp.forward(x1_train,x2_train,x3_train)
    y_pred_test = mlp.forward( x1_test, x2_test, x3_test)
    y_pred_plot= mlp.forward(x1_plot,x2_plot,x3_plot)

    if np.any(np.isnan(y_pred_train)) or np.any(np.isinf(y_pred_train)) or \
       np.any(np.isnan(y_pred_test)) or np.any(np.isinf(y_pred_test)):
        print("Warning: Predictions contain invalid values. Skipping update.")
        return (scatter, scatter_pred, loss_line, val_line, w1_line, w2_line, w3_line,w4_line,w5_line,w6_line, b_line)

    errors = (y_train - y_pred_train)
    val_errors =(y_test - y_pred_test)
    errors_cpu = cp.asnumpy(errors) if isinstance(errors, cp.ndarray) else errors
    val_errors_cpu = cp.asnumpy(val_errors) if isinstance(val_errors, cp.ndarray) else val_errors

    # Compute gradients
    grad_w1 = -2 * np.mean(errors * x1_train)
    grad_w2 = -2 * np.mean(errors * x2_train)
    grad_w3 = -2 * np.mean(errors * x3_train)
    grad_w4 = -2 * np.mean(errors * np.cos(x3_train))
    grad_w5 = -2 * np.mean(errors * np.sin(x2_train))
    grad_w6 = -2 * np.mean(errors * (x1_train**2))
    grad_b = -2 * np.mean(errors)
    grads = np.array([grad_w1, grad_w2, grad_w3, grad_w4, grad_w5, grad_w6, grad_b])
    if np.any(np.isnan(grads)) or np.any(np.isinf(grads)):
        print("Warning: Gradients contain invalid values. Skipping update.")
        return (scatter, scatter_pred, loss_line, val_line, w1_line, w2_line, w3_line,w4_line,w5_line,w6_line, b_line)

    # Update weights
    delta_w1 = lr * grad_w1
    delta_w2 = lr * grad_w2
    delta_w3 = lr * grad_w3
    delta_w4=lr * grad_w4
    delta_w5=lr * grad_w5
    delta_w6=lr * grad_w6
    delta_b = lr * grad_b
    
    w1 -= delta_w1
    w2 -= delta_w2
    w3 -= delta_w3
    w4-= delta_w4
    w5-= delta_w5
    w6-= delta_w6
    b -= delta_b

    weights= np.array([w1,w2,w3,w4,w5,w6,b])

    w1, w2, w3,w4,w5,w6, b = np.clip(weights, -10, 10)

    # Compute losses
    mse_train = mlp.backward(x1_train,x2_train,x3_train,y_train,lr)
    mse_val = np.mean(val_errors ** 2)
    loss_values.append(float(mse_train))
    val_loss_values.append(float(mse_val))
    w1_history.append(float(w1))
    w2_history.append(float(w2))
    w3_history.append(float(w3))
    w4_history.append(float(w4))
    w5_history.append(float(w5))
    w6_history.append(float(w6))
    b_history.append(float(b))
    epoch_values.append(float(epoch))

    

    # Update plots
    
    loss_line.set_data(epoch_values, loss_values)
    val_line.set_data(epoch_values, val_loss_values)
    w1_line.set_data(epoch_values, w1_history)
    w2_line.set_data(epoch_values, w2_history)
    w3_line.set_data(epoch_values, w3_history)
    w4_line.set_data(epoch_values, w4_history)
    w5_line.set_data(epoch_values, w5_history)
    w6_line.set_data(epoch_values, w6_history)
    b_line.set_data(epoch_values, b_history)

    w1_line.set_label(f"W1: {w1:.4f}")
    w2_line.set_label(f"W2: {w2:.4f}")
    w3_line.set_label(f"W3: {w3:.4f}")
    w4_line.set_label(f"W4: {w4:.4f}")
    w5_line.set_label(f"W5: {w5:.4f}")
    w6_line.set_label(f"W6: {w6:.4f}")
    b_line.set_label(f"Bias: {b:.4f}")
    
    

    #Update predicitons
    global surface_pred
    for artist in ax.collections[:]: 
      if artist is surface_pred:  
        artist.remove()


    thread = threading.Thread(target=compute_pred_mesh)
    thread.start()
    thread.join()
    
    if plotted_surface[0] is not None:
       plotted_surface[0].remove()

    plotted_surface[0]=ax.plot_surface(x1_mesh.get(), x2_mesh.get(), y_pred_mesh.get(), color='lime', alpha=0.3)
    if np.any(np.isnan(y_pred_mesh))or np.any(np.isinf(y_pred_mesh)):
        print("Warning: Sorted predictions contain invalid values.Skipping update")
        return (scatter,scatter_test,loss_line,val_line,w1_line,w2_line,w3_line,b_line)

    surface_pred = ax.plot_surface(x1_mesh.get(), x2_mesh.get(), y_pred_mesh.get(), color='lime', alpha=0.3, label="Prediction")
    ax.set_zlim(min(np.min(y_train).get(), np.min(y_pred_mesh).get()) - 1, max(np.max(y_train).get(), np.max(y_pred_mesh).get()) + 1) 


    # Update histogram
    ax3.clear()
    ax3.hist(errors_cpu, bins=30, color='red', alpha=0.7, label='Training Errors')
    ax3.hist(val_errors_cpu, bins=30, color='lime', alpha=0.7, label='Validation Errors')
    ax3.set_xlabel('Error')
    ax3.set_ylabel('Frequency')
    ax3.legend(loc='upper right')

    #Update chosen dependencies
    choice=radio.value_selected
    
    ax2d.clear()
    if choice == 'x1 vs y':
        if w1_true != 0:
            x_plot = np.linspace(np.min(x1_train), np.max(x1_train), 200)
            x_plot_cpu = x_plot.get()
            y_pred_plot = f_train(x_plot, np.mean(x2_train), np.mean(x3_train), w1, w2, w3, w4, w5, w6, b)
            ax2d.plot(x_plot_cpu, f_true(x_plot, np.mean(x2_train), np.mean(x3_train)).get(), label='True', color='hotpink', alpha=0.7)
            scatter_2d =ax2d.scatter(x1_train_cpu,y_train_cpu, label="Training Data", color='blue', marker='o')
            ax2d.plot(x_plot_cpu, y_pred_plot.get(), label='Predicted', color='lime', alpha=0.7)
            ax2d.set_xlabel("x1")
        else:
           print("x1 has not been defined. Let´s look at dependencies you defined! :)")
           if w2_true:
             radio.set_active(1) 
           else:
              radio.set_active(2) 

    elif choice == 'x2 vs y':
      if w2_true != 0:
        x_plot = np.linspace(np.min(x2_train), np.max(x2_train), 200)
        x_plot_cpu = x_plot.get()
        y_pred_plot = f_train(np.mean(x1_train), x_plot, np.mean(x3_train), w1, w2, w3, w4, w5, w6, b)
        ax2d.plot(x_plot_cpu, f_true(np.mean(x1_train), x_plot, np.mean(x3_train)).get(), label='True', color='hotpink', alpha=0.7)
        scatter_2d =ax2d.scatter(x2_train_cpu,y_train_cpu, label="Training Data", color='blue', marker='o')
        ax2d.plot(x_plot_cpu, y_pred_plot.get(), label='Predicted', color='lime', alpha=0.7)
        ax2d.set_xlabel("x2")
      else:
         print("x2 has not been defined. Let´s look at dependencies you defined! :)")
         radio.set_active(1)

    elif choice == 'x3 vs y':
        if w3_true != 0:
            x_plot = np.linspace(np.min(x3_train), np.max(x3_train), 200)
            x_plot_cpu = x_plot.get()
            y_pred_plot = f_train(np.mean(x1_train), np.mean(x2_train), x_plot, w1, w2, w3, w4, w5, w6, b)
            ax2d.plot(x_plot_cpu, f_true(np.mean(x1_train), np.mean(x2_train), x_plot).get(), label='True', color='hotpink', alpha=0.7)
            scatter_2d =ax2d.scatter(x3_train_cpu,y_train_cpu, label="Training Data", color='blue', marker='o')
            ax2d.plot(x_plot_cpu, y_pred_plot.get(), label='Predicted', color='lime', alpha=0.7)
            ax2d.set_xlabel("x3")
        else: 
           print("x3 has not been defined. Let´s look at dependencies you defined! :)")
           radio.set_active(0) 

    ax2d.set_ylabel("y")
    ax2d.grid(True, linestyle="--", alpha=0.5)
    ax2d.legend(loc="upper left")
    



    # Adjust axes limits
    ax.set_ylim(min(numpy.min(y_train.get()), numpy.min(y_pred_plot.get())) - 1, max(numpy.max(y_train.get()), numpy.max(y_pred_plot.get())) + 1)
    ax2.set_xlim(0, len(loss_values))  # Ensure x-axis scales with epochs
    if loss_values:
      ylim_max = max(max(loss_values), max(val_loss_values)) * 1.1
    else:
     ylim_max = 1
    ax2.set_ylim(0, ylim_max)
    if w1_history:
      ylim_min = min(min(w1_history), min(w2_history), min(w3_history), min(b_history)) - 0.5
      ylim_max_w = max(max(w1_history), max(w2_history), max(w3_history), max(b_history)) + 0.5
    else:
     ylim_min, ylim_max_w = -1, 1
    ax4.set_ylim(ylim_min, ylim_max_w)

    # Update learning rate
    if all(abs(d) < 0.001 for d in [delta_w1, delta_w2, delta_w3, delta_w4,delta_w5,delta_w6,delta_b]) and lr > 0.0001:
        w1_slider.set_val(lr * decay_factor)

    # Update legends
    w1_line.set_label(f"W1: {w1:.4f}")
    w2_line.set_label(f"W2: {w2:.4f}")
    w3_line.set_label(f"W3: {w3:.4f}")
    w4_line.set_label(f"W4: {w4:.4f}")
    w5_line.set_label(f"W5: {w5:.4f}")
    w6_line.set_label(f"W6: {w6:.4f}")
    b_line.set_label(f"Bias: {b:.4f}")
    ax4.legend(loc='upper left')
    loss_line.set_label(f"Loss: {mse_train:.4f}")
    val_line.set_label(f"Validation Loss: {mse_val:.4f}")
    ax2.legend(loc='upper right')
    fig.canvas.draw_idle() 
    return (scatter, scatter_pred, loss_line, val_line, w1_line, w2_line, w3_line,w4_line,w5_line,w6_line, b_line,twod_pred,twod_true, plotted_surface[0])


# Animation
ani = FuncAnimation(fig, update, frames=None, interval=50, blit=False, repeat=False,cache_frame_data=False)
plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
plt.show()