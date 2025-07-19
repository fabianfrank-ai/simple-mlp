## Simple MLP: Multivariate Polynomial and Non-Linear Regression Visualization

## Project Goal

The simple-mlp project aims to provide an intuitive, real-time visualization of a neural network (Multilayer Perceptron, MLP) learning to approximate a user-defined multivariate function, such as a third-degree polynomial or non-linear expressions involving trigonometric and quadratic terms. By implementing manual gradient descent and symbolic computation, the project serves as a didactic tool to illustrate the mechanics of model fitting, gradient-based optimization, bias-variance trade-offs, and error behavior. The interactive visualizations make the optimization process (loss reduction, weight updates, and error distributions) accessible and understandable, particularly for educational purposes.

## Methodology & Implementation

## Data





Synthetic Data: The project generates synthetic samples of the form y = f(x1, x2, x3) + ε, where f is a user-defined function (e.g., x1 + x2 + x3 + 2 or sin(x3) + cos(x2) + x1^2) and ε is normally distributed noise to simulate real-world variability.



Input Features: Supports three input variables (x1, x2, x3), with random values drawn from a uniform distribution ([-2, 2]).



Feature Engineering: The model includes linear terms (x1, x2, x3), trigonometric terms (cos(x3), sin(x2)), and quadratic terms (x1^2) based on the user-defined function.

Train/Test Split





Data is split into training (80%) and test (20%) sets using scikit-learn's train_test_split to monitor generalization and detect overfitting.

## Model





## Architecture: A simple MLP implemented in symbolicMLP (assumed custom module) with seven learnable parameters:





w1, w2, w3: Weights for linear terms (x1, x2, x3).



w4, w5: Weights for trigonometric terms (cos(x3), sin(x2)).



w6: Weight for quadratic term (x1^2).



b: Bias term.



Function Representation: The model approximates the user-defined function using symbolic expressions parsed via SymPy, allowing flexibility for linear and non-linear terms.

## Optimization





Loss Function: Mean Squared Error (MSE) for both training and validation data.



Gradient Computation: Gradients are computed manually for each parameter using the chain rule, ensuring transparency in the optimization process.



Gradient Descent: Manual implementation of gradient descent with:





Adaptive learning rate controlled via an interactive slider (range: 0.0001 to 0.01).



Decay mechanism: Learning rate is halved when parameter updates fall below 0.001 to improve convergence.



Weight clipping to [-10, 10] to prevent exploding gradients.



Error Handling: Checks for NaN/Inf values in predictions and gradients, skipping updates if detected to maintain stability.

Visualization

The project uses Matplotlib with Tkinter for real-time, interactive visualizations:





## 3D Plot:





Displays training and test data as scatter points.



Shows true function and predicted function as surfaces, updated per epoch.



Axes dynamically adjust based on user-selected visualization variables (x1, x2, or x3).



## 2D Plots:





Loss Plot: Tracks training and validation loss, with weight trajectories (w1, w2, w3, w4, w5, w6, b) on a secondary axis.



Prediction Plot: Shows true vs. predicted values for a selected variable (x1 vs y, x2 vs y, or x3 vs y), chosen via radio buttons.



Error Histogram: Displays the distribution of training and validation residuals.



## Interactivity:





Learning Rate Slider: Adjusts the learning rate in real-time.



Radio Buttons: Allow switching between visualization modes (x1 vs y, x2 vs y, x3 vs y).



Dynamic Scaling: Axes automatically adjust to accommodate evolving loss, weights, and predictions.



Note: The prediction history plot (overlaying previous predictions) is currently disabled.

## Key Features





Real-Time Animation: Visualizes the learning process, including prediction surfaces, loss, weights, and error distributions, updated every epoch.



Manual Gradient Descent: Implements optimization without relying on deep learning frameworks, enhancing understanding of the underlying mechanics.



Symbolic Computation: Uses SymPy to parse user-defined functions, supporting linear, trigonometric, and quadratic terms.



GPU/CPU Flexibility: Leverages CuPy for GPU acceleration when available, with a seamless fallback to NumPy for CPU-based computation.



Interactive Controls: Includes sliders and radio buttons for dynamic adjustment of learning rate and visualization modes.



Educational Focus: Designed as a didactic tool to illustrate model fitting, optimization, and bias-variance trade-offs.

Tech Stack





NumPy/CuPy: Vectorized mathematical computations, with CuPy for GPU acceleration.



Matplotlib: Real-time, multi-panel plotting for 3D and 2D visualizations.



Tkinter: GUI integration for interactive sliders and radio buttons.



SymPy: Symbolic computation for parsing and evaluating user-defined functions.



scikit-learn: Data splitting for training and test sets.



Threading: Non-blocking computation of predicted surfaces.

Prerequisites





Python 3.8 or higher



## Required packages:





numpy



matplotlib



sympy



scikit-learn



cupy-cudaXX (optional, replace XX with your CUDA version, e.g., cupy-cuda12x)



Tkinter (typically included with Python; install separately if needed):





Ubuntu/Debian: sudo apt-get install python3-tk



macOS (Homebrew): brew install python-tk



CUDA toolkit (for CuPy/GPU support, optional)

## Limitations





Function Support: Limited to expressions with x1, x2, x3, sin(x3), cos(x2), and x1^2. Extending to other functions (e.g., tan, exp) requires updating the terms dictionary.



MLP Implementation: Relies on symbolicMLP, which is not provided. Users must ensure compatibility with CuPy/NumPy.



Visualization: 3D plots may be cluttered for complex functions. Prediction history plotting is currently disabled.



Performance: CPU fallback (NumPy) is slower than GPU (CuPy) for large datasets.



Threading: Basic threading for surface computation may not be optimal on all systems.

## Future Ideas & Extensions





Regularization: Implement L1/L2 regularization to prevent overfitting.



Comparison: Benchmark against scikit-learn’s regression models or closed-form solutions (e.g., normal equation).



Multivariate Expansion: Generalize to arbitrary input dimensions (x1, x2, ..., xn).



Real-World Datasets: Support loading and processing real datasets (e.g., CSV files).



Enhanced GUI: Add sliders for additional hyperparameters (e.g., polynomial degree, noise level) and buttons to pause/resume training.



Function Support: Expand to include more mathematical functions (tan, exp, log, etc.) via SymPy.



Prediction History: Re-enable overlaying previous prediction curves for visual comparison.



Optimization Enhancements: Explore momentum-based gradient descent or adaptive optimizers (e.g., Adam).





