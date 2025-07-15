# simple-mlp

## Project Goal

This project provides a visualization of a third degree multivariate polynomal regression trained using manual gradient descent. The goal is to illustrate how the model learns to approximate a non-linear target function over time, while simultaneaously making the underlying optimization process (loss reduction,weight updates, error behaviour) inuitively understandable through real-time visualizations.

## Methodology & Implementation
-Data: Synthetic samples of the form y = w₃·x³ + w₂·x² + w₁·x + b + ϵ,
where ϵ is normally distributed noise. Changes to expanding the form to more exponents, root functions, sin,tan,cos and more are planned and scheduled in order to provide a more realistic experience regarding given data.

-Feature Engineering:  Polynomial features up to degree 3: x, x², x³. For now.

-Train/Test Split: Data is split to monitor generalization and overfitting

-Model: Simple model with four learnable parameters: w1, w2, w3 and b

-Optimization: 
   -Custom implementation of Mean Squared Error (MSE) and its gradients
   -Manual gradient descent with adaptive learning rates
   -Decay mechanisms if parameter updates become minimal

-Visualization
     -Main Plot: Predicted curve vs true curve(updated every epoch)
     -Loss Plot: Training and validation loss across epochs
     -Weight Trajectories: Evolution of w1,w2,w3 and b
     -Error Histogram: Distribution of residuals for train/test sets
     (-Prediction History. All previous prediction curves overlaid --currently out of order)

## Key Features 

Intuitive real-time animation of the learning process
Fully manual gradient computation- no deep learning libraries involved (for the sake of learning and simplicity)
Dynamically scaled axes and automated learning rate adjustmens
Didactic tool for understanding model fitting, bias variance trade-offs, and the mechanics of gradient descent

## Tech Stack

numpy- vectorized mathematical computations
matplotlib-real-time, multi-panel plotting

## Future Ideas & Extensions

-L1/L2 regularization
-Compare with scikit-learn or closed-form solutions
-Generalize to multivariable input (x1,x2,x3...)
-Try real-world datasets
-GUI-sliders to interactively tweak hyperparameters(learning rate,degree,etc)



