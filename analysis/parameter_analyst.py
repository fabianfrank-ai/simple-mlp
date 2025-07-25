import pandas as pd
import numpy as np
import cupy as cp
import os
import matplotlib.pyplot as plt
from datetime import datetime
import json


class parameter_analysis:
    def __init__(self,weights, bias, lr, l2_lambda, hidden_layers, epochs, batch_size,solver,activation_function,loss,val_loss):
        self.weights = weights
        self.bias = bias
        self.lr = lr
        self.l2_lambda = l2_lambda
        self.hidden_layers = hidden_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.solver = solver
        self.activation_function = activation_function
        self.loss = loss
        self.val_loss = val_loss

        self.used_epochs =0
        self.deleted_epochs=0
        


    def calculate_weight_changes(self):
        epoch_weight_changes = []
        layer_weight_changes = []
        if len(self.weights) <= 1: 
            return 0,0
        else:
            for epoch in  range(1, len(self.weights)):
                current_weights = self.weights[epoch]
                previous_weights = self.weights[epoch-1]
                weight_change=current_weights - previous_weights
                weight_change_avg=np.mean(np.abs(weight_change))
                relative_epoch_change=weight_change_avg/np.mean(np.abs(previous_weights))
                if relative_epoch_change > 0.01 or epoch<=20:
                    epoch_weight_changes.append(weight_change_avg)
                    self.used_epochs += 1
                    epoch_layer_changes=[]
                    for layer in range(len(current_weights)):
                        current_weights_layer = current_weights[layer]
                        previous_weights_layer = previous_weights[layer]
                        weight_change_layer = current_weights_layer - previous_weights_layer
                        weight_change_layer_avg = np.mean(np.abs(weight_change_layer))
                        epoch_layer_changes.append(weight_change_layer_avg)
                    layer_weight_changes.append(epoch_layer_changes)
                    self.deleted_epochs += 1
     
        return epoch_weight_changes, layer_weight_changes
    
    def generate_training_report(self):
        epoch_weight_changes, layer_weight_changes = self.calculate_weight_changes()
        loss_changes_avg, val_loss_changes_avg = self.loss_trend_analysis()
        bias_changes_avg = self.bias_changes_analysis()

        report = "\n📊 Training report\n"
        report += f"– Epochs with relevant weight adjustments (>1%): {self.used_epochs} out of {self.epochs}\n"
        report += f"– Unchanged epochs (<1%): {self.deleted_epochs}\n"
        report += f"– Average weight change per epoch: {np.mean(epoch_weight_changes):.4f}\n"
        report += f"– Average layer change per epoch: {np.mean(layer_weight_changes):.4f}\n"
        report += f"– Average loss change: {np.mean(loss_changes_avg):+.4f}\n"
        report += f"– Average validation loss change: {np.mean(val_loss_changes_avg):+.4f}\n\n"
        report += f"Average bias change: {np.mean(bias_changes_avg):.4f}\n\n"

        # Interpretation
        report += "🧠 INTERPRETATION\n"

        if np.mean(val_loss_changes_avg) > 0:
            report += "→ The validation error tends to increase → Overfitting likely.\n"
        else:
            report += "→ The validation error decreases or remains constant → Model generalizes well.\n"

        if np.mean(epoch_weight_changes) < 0.001:
            report += "→ Very small weight changes → Learning rate might be too low.\n"
        elif np.mean(epoch_weight_changes) > 0.1:
            report += "→ Very high weight fluctuation → Learning rate might be too high.\n"
        else:
            report += "→ Weight changes are within the expected range.\n"

        if self.lr > 0.05:
            report += f"→ Learning rate ({self.lr}) is relatively high → Possible instability.\n"
        elif self.lr < 0.0001:
            report += f"→ Learning rate ({self.lr}) is very low → Slow learning likely.\n"

        if np.mean(val_loss_changes_avg) > 0 and np.mean(epoch_weight_changes) > 0.1:
           report += "\n📌 Diagnosis: Model shows instability and is in danger of overfitting.\n"
        elif np.mean(epoch_weight_changes) < 0.001:
            report += "\n📌 Diagnoseis: Learning process is slow.\n"
        else:
            report += "\n📌 Diagnosis: Training is stable and working normally.\n"

        return report


    def loss_trend_analysis(self):
        loss_changes_avg=[]
        val_loss_changes_avg=[]
        for epoch in range(1,len(self.loss)):
            loss_change=self.loss[epoch] - self.loss[epoch-1]
            val_loss_change=self.val_loss[epoch] - self.val_loss[epoch-1]
            if abs(loss_change) > 0.01 or abs(val_loss_change) > 0.01 or epoch<=20:
                loss_changes_avg.append(np.abs(loss_change))
                val_loss_changes_avg.append(np.abs(val_loss_change))
        return loss_changes_avg, val_loss_changes_avg
    
    def bias_changes_analysis(self):
        bias_changes_avg=[]
        for epoch in range(len(self.bias)):
            bias_change=self.bias[epoch] - self.bias[epoch-1]
            if abs(bias_change) > 0.01 or epoch<=20:
                bias_changes_avg.append(np.mean(np.abs(bias_change)))
        return bias_changes_avg

        
    def weight_analysis(self):
        pass
    def bias_analysis(self):
        pass

    def lr_impact_analysis(self):
        pass
    def regularization_impact_analysis(self):
        pass
    
    

    