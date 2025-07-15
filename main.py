import numpy as np 
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.model_selection import train_test_split
from matplotlib.widgets import Slider, Button




fig, (ax,ax2, ax3)=plt.subplots(3,1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 3,1]})
lines= []


ax.grid(True, linestyle='--', alpha=0.5)
ax.set_xlabel('x', fontsize=14)
ax.set_ylabel('y', fontsize=14)
ax.set_title("Training in Action", fontsize=16)
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.set_title("Loss, Validation Loss and weights", fontsize=16)
ax3.grid(True, linestyle='--', alpha=0.5)
ax3.set_title("Prediction History", fontsize=16)
ax4=ax2.twinx()



text = ax.text(0.5, 1.05, '', transform=ax.transAxes, ha='left', fontsize=14)


fill_patches = []
y_history = []  
loss_values = []  # Store loss values for each epoch
epoch_values = []  # Store epoch values for plotting
predictions = []  # Store predictions for each epoch
val_loss_values = []  # Store validation loss values for each epoch
w1_history = []  # Store w1 values for each epoch
w2_history = []  # Store w2 values for each epoch   
w3_history = []  # Store w3 values for each epoch
b_history = []  # Store bias values for each epoch

# Generate data
x=np.random.randn(100,)
#x= (x-np.mean(x))/np.std(x) * 2 + 5  # Normalize and shift to range [3, 7]
x_sorted= np.sort(x,axis=0)
x_sorted=x_sorted

x1=x**1
x2=x**2
x3=x**3

x_plot=np.linspace(np.min(x),np.max(x),100)
x1_plot=x_plot
x2_plot=x_plot**2
x3_plot=x_plot**3

X= np.stack([x1,x2,x3],axis=1)

sort_index= np.argsort(x1)





x1_sorted=x1[sort_index]
x2_sorted=x2[sort_index]
x3_sorted=x3[sort_index]

y=x3+x2+x1+2+np.random.randn(100,)
#y=(y-np.mean(y))/np.std(y) * 2 + 1  # Normalize and shift to range [0, 3]
y_true=x3_plot+x2_plot+x1_plot+2
y_true_sorted=y_true[sort_index]

x_train, x_test, y_train, y_test= train_test_split(X,y,test_size=0.2, random_state=42)


# Initialize weights and biases
w1=np.random.randn()
w2=np.random.randn()
w3=np.random.randn()

b=np.random.randn()

#Calculate predictions
y_pred=w3*x3+w2*x2+w1*x1+b
y_pred_sorted=w3*x3_sorted+w2*x2_sorted+w1*x1_sorted+b
y_val=x3_plot+x2_plot+x1_plot+2
y_plot=w3*x3_plot+w2*x2_plot+w1*x1_plot+b

#MSE
summe= 0
summe_w1=0
summe_w2=0
summe_w3=0

ax.set_xlim(np.min(x1) - 0.5, np.max(x1) + 0.5)
ax.set_ylim(np.min(y) - 1, np.max(y) + 1)


summe_b=0
liste=[]
# Calculate the initial MSE and gradients
for i in range(len(x_train)):
    y_pred = w3*x_train[i,2] + w2*x_train[i,1] + w1*x_train[i,0] + b
    fehler = (y_train[i] - y_pred)**2
    summe = summe + fehler
    ableitung_w1 = (y_train[i] - y_pred) * x_train[i,0]
    ableitung_w2 = (y_train[i] - y_pred) * x_train[i,1]
    ableitung_w3 = (y_train[i] - y_pred) * x_train[i,2]
    summe_w1 = summe_w1 + ableitung_w1
    summe_w2 = summe_w2 + ableitung_w2
    summe_w3 = summe_w3 + ableitung_w3

    ableitung_b = (y_train[i] - y_pred)
    summe_b = summe_b + ableitung_b

# Calculate the initial MSE
MSE=summe/len(x_train)
ableitung1_fehler=summe_w1*(-2/100)
ableitung2_fehler=summe_w2*(-2/100)
ableitung3_fehler=summe_w3*(-2/100)

yachsabschnitt_fehler=summe_b*(-2/100)

w1=w1-ableitung1_fehler*0.001
w2=w2-ableitung2_fehler*0.001
w3=w3-ableitung3_fehler*0.001


b=b-yachsabschnitt_fehler*0.001

loss_line, = ax2.plot([], [], color='purple', label='Loss')
val_line, = ax2.plot([], [], color='orange', label='Validation Loss')
ax2.legend(loc='upper right')

w1_line, = ax4.plot([], [], label=[f"W1: {w1:.4f}"], color='green', zorder=7, linestyle='--',alpha=0.3)
w2_line, = ax4.plot([], [], label=[f"W2: {w2:.4f}"], color='red', zorder=7,linestyle='--',alpha=0.3) 
w3_line, = ax4.plot([], [], label=[f"W3: {w3:.4f}"], color='blue', zorder=7, linestyle='--',alpha=0.3)
b_line, = ax4.plot([], [], label=[f"Bias:{b:.4f}"], color='black', zorder=7, linestyle='--',alpha=0.3)

# Create a scatter plot of the training data
scatter = ax.scatter(x, y, zorder=2, label="Scattered training data", color='blue',marker='o')
#create a scatter plot of the test data
scatter_test = ax.scatter(x_test[:, 0], y_test, zorder=2, label="Scattered test data", color='cyan',marker='x')
scatter_train = ax.scatter(x_train[:, 0], y_train, zorder=2, label="Scattered train data", color='hotpink',marker='x')



# Plot the true graph
line2, = ax.plot(x_plot, y_true, label="True Graph", color='red', zorder=3,linewidth=1.5)
# Plot the first prediction
line3, = ax.plot(x1_plot, y_plot, label="First Prediction", color='orange', zorder=5,linestyle='--')
#Progress line
line, = ax.plot([], [], color='lime',label='Current Prediction', zorder=6, linewidth=2) 



ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend(loc='upper right')
ax.legend(loc='upper left')
ax3.set_xlabel('Error')
ax3.set_ylabel('Frequency')
ax3.legend(loc='upper right')
ax4.set_ylabel('Weights')
ax3.legend(loc='upper right')

xmin=np.min(x)
xmax=np.max(x)
u=1000
v=0


x1_train = x_train[:, 0]
x2_train = x_train[:, 1]
x3_train = x_train[:, 2]

num_epochs = len(x) 

fill = None
# Function to update the plot
def update(epoch):
    global line, w1, w2, w3, b, v, fill_patches, y_sorted, fill

    y_pred_val = w3 * x_test[:, 2] + w2 * x_test[:, 1] + w1 * x_test[:, 0] + b

    v += 1

    # Learning rates and decay
    #lr_w1 = fig.add_axes([0.15, 0.05, 0.1, 0.03])
    #w1_slider= plt.Slider(lr_w1, 'LR W1', 0.0001, 0.01, valinit=0.001, orientation='horizontal')
    lr_w1 =0.001 #w1_slider.val
    lr_w2 = 0.001
    lr_w3 = 0.001
    lr_b = 0.004
    decay_factor = 0.5

    old_w1, old_w2, old_w3, old_b = w1, w2, w3, b

    # Vectorized predictions and gradients
    y_pred_train = w3 * x_train[:, 2] + w2 * x_train[:, 1] + w1 * x_train[:, 0] + b
    errors = y_train - y_pred_train
    val_errors=y_test-y_pred_val
    # Calculate the sum of squared errors and gradients
    summe = np.sum(errors ** 2)
    summe_w1 = np.sum(errors * x_train[:, 0])
    summe_w2 = np.sum(errors * x_train[:, 1])
    summe_w3 = np.sum(errors * x_train[:, 2])
    summe_b = np.sum(errors)

    # Gradients
    ableitung1_fehler = summe_w1 * (-2 / len(x_train))
    ableitung2_fehler = summe_w2 * (-2 / len(x_train))
    ableitung3_fehler = summe_w3 * (-2 / len(x_train))
    yachsabschnitt_fehler = summe_b * (-2 / len(x_train))

    # Update weights and bias every epoch
    w1 -= ableitung1_fehler * lr_w1
    w2 -= ableitung2_fehler * lr_w2
    w3 -= ableitung3_fehler * lr_w3
    b -= yachsabschnitt_fehler * lr_b

    b = np.clip(b, -10, 10)  # Prevent extreme values

    # Update loss
    MSE_training = summe / len(x_train)
    val_loss = np.mean(val_errors ** 2) 
     # Store loss values for plotting
    loss_values.append(MSE_training)
    epoch_values.append(epoch)
    val_loss_values.append(val_loss)
    b_history.append(b)

    # Update predictions for plotting
    sort_idx = np.argsort(x1_train)
    x1_sorted = x1_train[sort_idx]
    x2_sorted = x2_train[sort_idx]
    x3_sorted = x3_train[sort_idx]
    y_sorted = w3 * x3_sorted + w2 * x2_sorted + w1 * x1_sorted + b

    # Store the current prediction for history
    y_history.append(y_sorted.copy())
    predictions.append(y_sorted)
    w1_history.append(w1)
    w2_history.append(w2)   
    w3_history.append(w3)
    # Calculate the errors for the histogram
    delta_w1 = np.abs(w1 - old_w1)
    delta_w2 = np.abs(w2 - old_w2)  
    delta_w3 = np.abs(w3 - old_w3)
    delta_b = np.abs(b - old_b)

    threshold = 0.001
    if delta_w1 < threshold and lr_w1 > 0.0001:
        lr_w1 = lr_w1*decay_factor # Reduce learning rate if change is small
        
    if delta_w2 < threshold and lr_w2 > 0.0001:    
        lr_w2 = lr_w2*decay_factor # Reduce learning rate if change is small
        
    if delta_w3 < threshold and lr_w3 > 0.0001:
        lr_w3 = lr_w3*decay_factor # Reduce learning rate if change is small
        

    if delta_b < threshold:
        lr_b = lr_b * decay_factor  # Reduce learning rate if change is small
       

    
   
    # Update loss line and scale x-axis with epochs
    loss_line.set_data(range(len(loss_values)), loss_values)
    val_line.set_data(range(len(val_loss_values)), val_loss_values)
    w1_line.set_data(range(len(loss_values)),w1_history )
    w2_line.set_data(range(len(loss_values)),w2_history)
    w3_line.set_data(range(len(loss_values)), w3_history)
    b_line.set_data(range(len(loss_values)), b_history)

   
    ax2.set_xlim(0, len(loss_values))  # Dynamically scale x-axis with epochs
    
    

    ax3.clear()  # Clear the previous plot
    ax3.set_xlabel('Error')
    ax3.set_ylabel('Frequency')
    ax3.legend(loc='upper right')
    ax3.hist(errors, bins=30, color='red', alpha=0.7, label='Training Errors')
    ax3.hist(val_errors, bins=30, color='lime', alpha=0.7, label='Validation Errors')
    ax3.legend(loc='upper right')
    ax4.legend(loc='upper left')

    ax4.clear()  # Clear the previous plot
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Weights')
    ax4.plot(range(len(loss_values)), w1_history, label=f"W1: {w1:.4f}", color='green', zorder=7, linestyle='--',alpha=0.3)
    ax4.plot(range(len(loss_values)), w2_history, label=f"W2: { w2:.4f}", color='red', zorder=7, linestyle='--',alpha=0.3)
    ax4.plot(range(len(loss_values)), w3_history, label=f"W3: {w3:.4f}", color='blue', zorder=7, linestyle='--',alpha=0.3)
    ax4.plot(range(len(loss_values)), b_history, label=f"Bias: {b:.4f}", color='black', zorder=7, linestyle='--',alpha=0.3)
    ax4.legend(loc='upper left')

    ax2.clear()  # Clear the previous plot
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.plot(range(len(loss_values)), loss_values, color='purple', label=[f'Loss: {MSE_training:.4f}'], zorder=4)
    ax2.plot(range(len(val_loss_values)), val_loss_values, color='orange', label=[f'Validation Loss: {val_loss:.4f}'], zorder=5)
    ax2.legend(loc='upper right')

    
    if max(val_loss_values)>max(loss_values):
        ax2.set_ylim(0, max(val_loss_values) * 1.1 if loss_values else 1)
    else:
        ax2.set_ylim(0, max(loss_values) * 1.1 if loss_values else 1)

   
   
    scatter_pred.set_offsets(np.c_[x1_sorted, y_sorted+np.random.randn(80,)])  # Update scatter plot with new predictions 
    
    ax.set_xlim(np.min(x1_train) - 0.5, np.max(x1_train) + 0.5)
    ax.set_ylim(np.min(y_train) - 1, np.max(y_train) + 1)


    ax4.set_ylim(np.min([w1, w2, w3,b]) - 0.5, np.max([w1, w2, w3,b]) + 0.5)

    # Update main prediction line
    line.set_data(x1_sorted, y_sorted)

    # Return all updated artists as a tuple
    return (line, loss_line,)
    

 # Sort the training data based on x1_train
sort_idx = np.argsort(x1_train)
x1_sorted = x1_train[sort_idx]
x2_sorted = x2_train[sort_idx]
x3_sorted = x3_train[sort_idx]
y_res = w3*x3_sorted+w2*x2_sorted+w1*x1_sorted+b   
    
    
 # Sort the training data for plotting
sort_idx = np.argsort(x1_train)
x1_sorted = x1_train[sort_idx]
x2_sorted = x2_train[sort_idx]
x3_sorted = x3_train[sort_idx]
y_res = w3*x3_sorted+w2*x2_sorted+w1*x1_sorted+b+np.random.randn(80,)

scatter_pred = ax.scatter(x1_sorted, y_res, zorder=2, label="Scattered prediction", color='red',marker='x')


# Calculate the initial MSE after the first update
summe_angepasst=0
for i in range(len(x_train)):
    y_pred2=w3*x_train[i,2]+w2*x_train[i,1]+w1*x_train[i,0]+b
    fehler=(y_train[i]-y_pred2)**2
    summe_angepasst=summe_angepasst+fehler

MSE_angepasst=summe_angepasst/len(x_train)

# Calculate the errors
print("MSE:", MSE)
print("Fehler Ableitung:", ableitung1_fehler )
print("Fehler Y-Achsenabschnitt:", yachsabschnitt_fehler)
print("MSE nach anpassung:", MSE_angepasst)
print("Gelerntes B:", b)
print("Gelerntes w" , w1, w2, w3)




ani = FuncAnimation(fig, update, frames=None, interval=50,blit=False, repeat=False)
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.title("Training in action")
plt.show()

