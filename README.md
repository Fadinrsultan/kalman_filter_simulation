
# 🎯 Kalman Filter Simulation – Projectile Tracking Under Measurement Dropouts

### State Estimation Using Kalman Filtering, Gaussian Process Regression, Ensemble Prediction, and RTS Smoothing

This project investigates **Kalman filtering and state estimation techniques** for tracking projectile motion under noisy and missing sensor measurements.

The simulation models a projectile moving in two-dimensional space under gravitational acceleration. Its objective is to estimate the projectile's position and velocity, even when sensor observations become temporarily unavailable.

Two approaches are implemented and evaluated:

1. **Kalman Filter with Ensemble Pseudo-Measurements**
2. **Kalman Filter with Rauch–Tung–Striebel (RTS) Smoothing**

The first method integrates Gaussian Process Regression (GPR), polynomial regression, and linear regression to estimate missing measurements.

The second method combines forward Kalman filtering with backward RTS smoothing to improve trajectory reconstruction.

The project includes simulation, prediction, measurement updates, uncertainty modeling, accuracy evaluation, and trajectory visualization.

## 🚀 Open in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Fadinrsultan/kalman_filter_simulation/blob/main/kalman_filter_simulation.ipynb)

**GitHub Repository:**

https://github.com/Fadinrsultan/kalman_filter_simulation

## 🎯 Project Objectives

The main objectives are to:

- Simulate two-dimensional projectile motion.
- Implement a Kalman filter for position and velocity estimation.
- Model noisy and intermittent sensor measurements.
- Handle extended measurement dropout intervals.
- Generate pseudo-measurements using regression models.
- Investigate adaptive uncertainty during missing observations.
- Implement Rauch–Tung–Striebel smoothing.
- Evaluate position and velocity estimation accuracy.
- Compare the behavior of two state-estimation approaches.
- Visualize true trajectories, noisy observations, and estimated states.

## ✨ Key Features

- **Projectile Motion Simulation:** Physics-based trajectory generation under gravity.
- **Kalman Filtering:** Recursive state estimation with uncertainty tracking.
- **Measurement Dropout Handling:** Supports intervals where observations are unavailable.
- **Gaussian Process Regression:** Predicts missing position measurements.
- **Polynomial Regression:** Models nonlinear trajectory behavior.
- **Linear Regression:** Provides an additional trajectory prediction model.
- **Ensemble Pseudo-Measurements:** Combines regression predictions during dropouts.
- **Adaptive Noise Scaling:** Adjusts uncertainty during extended observation gaps.
- **RTS Smoothing:** Refines state estimates using forward and backward information.
- **Position and Velocity Evaluation:** Calculates normalized estimation accuracy.
- **Trajectory Visualization:** Compares estimates with ground truth.

## 🧠 Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Simulation and implementation |
| NumPy | Matrix operations and numerical calculations |
| Matplotlib | Visualization |
| Scikit-learn | Regression models |
| GaussianProcessRegressor | Nonlinear trajectory prediction |
| RBF Kernel | Smooth function modeling |
| WhiteKernel | Observation noise modeling |
| LinearRegression | Linear prediction |
| Polynomial Regression | Nonlinear trajectory approximation |
| Kalman Filter | Recursive state estimation |
| RTS Smoother | Backward state refinement |
| Jupyter Notebook | Interactive experimentation |
| Google Colab | Notebook execution |

## 🏗️ System Architecture

The simulation compares two approaches for handling unavailable measurements.

```text id="h5k3n9"
                START
                  |
                  v
         Projectile Simulation
                  |
                  v
       Generate Ground Truth
                  |
                  v
       Generate Noisy Sensor
            Observations
                  |
                  v
       Introduce Measurement
             Dropouts
                  |
         +--------+--------+
         |                 |
         v                 v
      METHOD 1          METHOD 2
         |                 |
         v                 v
    Kalman Filter     Kalman Filter
         |                 |
         v                 v
    Measurement        Forward State
     Available?         Estimation
         |                 |
     +---+---+             v
     |       |        Store States
    Yes      No        & Covariances
     |       |             |
     v       v             v
   Normal  Ensemble      Backward RTS
   Update  Prediction      Smoothing
             |               |
        +----+----+          |
        |    |    |          |
        v    v    v          |
       GPR  Poly Linear     |
        |    |    |          |
        +----+----+          |
             |               |
             v               v
       Pseudo-Measurement  Smoothed
             Update        Estimates
             |               |
             +-------+-------+
                     |
                     v
             Accuracy Evaluation
                     |
                     v
            Trajectory Visualization
```

## 📐 Mathematical Model

The system models projectile motion in two-dimensional space.

### State Representation

The state vector is:

\[
X_t =
\begin{bmatrix}
x_t \\
y_t \\
v_{x,t} \\
v_{y,t}
\end{bmatrix}
\]

Where:

- `x`: Horizontal position
- `y`: Vertical position
- `vx`: Horizontal velocity
- `vy`: Vertical velocity

The system has four state variables.

### Motion Equations

Horizontal position:

\[
x_{t+1}=x_t+v_{x,t}\Delta t
\]

Vertical position:

\[
y_{t+1}=y_t+v_{y,t}\Delta t-\frac{1}{2}g\Delta t^2
\]

Horizontal velocity:

\[
v_{x,t+1}=v_{x,t}
\]

Vertical velocity:

\[
v_{y,t+1}=v_{y,t}-g\Delta t
\]

Where:

- `g = 9.81 m/s²`
- `Δt = 0.1 s`

The initial projectile conditions are:

| Parameter | Value |
|-----------|-------|
| Initial height | 10 m |
| Launch speed | 20 m/s |
| Launch angle | 45° |
| Gravity | 9.81 m/s² |
| Time step | 0.1 s |
| Simulation steps | 100 |

## 🔍 Kalman Filtering

The Kalman filter estimates hidden states using a prediction-update cycle.

### Prediction Step

The state is predicted using the system motion model:

\[
\hat{x}_{k|k-1}=A\hat{x}_{k-1|k-1}+Bu_k
\]

The covariance is predicted using:

\[
P_{k|k-1}=AP_{k-1|k-1}A^T+Q
\]

Where:

- `A` is the state transition matrix.
- `B` is the control-input matrix.
- `u` represents the gravitational input.
- `Q` is the process noise covariance.
- `P` is the state estimation covariance.

### Measurement Update

When a sensor observation is available, the Kalman gain is calculated:

\[
K_k=P_{k|k-1}H^T
(HP_{k|k-1}H^T+R)^{-1}
\]

The state estimate is corrected:

\[
\hat{x}_{k|k}
=
\hat{x}_{k|k-1}
+
K_k(z_k-H\hat{x}_{k|k-1})
\]

The covariance is also updated to reflect the new measurement information.

During sensor dropouts, the two methods use different strategies to maintain or improve state estimation.

## 🔬 Method 1: Kalman Filter with Ensemble Pseudo-Measurements

The first method uses an ensemble of regression models to generate estimated measurements during sensor dropouts.

### Concept

During normal operation, the Kalman filter updates its states using actual noisy position measurements.

When observations are missing, the algorithm predicts the projectile's position through regression models trained on historical data.

These predictions are combined into pseudo-measurements that support the Kalman filter.

### Regression Models

**1. Gaussian Process Regression (GPR)**

GPR models the relationship between time and projectile position.

The implementation uses Gaussian process kernels, including RBF and WhiteKernel components.

It provides nonlinear predictions based on available historical data.

**2. Polynomial Regression**

Polynomial regression approximates the projectile trajectory.

A second-degree polynomial is used:

\[
p(t)=a_0+a_1t+a_2t^2
\]

This model is particularly relevant to projectile motion, where vertical position follows a parabolic trajectory.

**3. Linear Regression**

Linear regression estimates future positions based on a linear relationship between time and observed motion.

### Ensemble Prediction

The regression model outputs are combined to generate pseudo-measurements during dropouts.

These pseudo-measurements provide an alternative input for the Kalman update step when real sensor information is unavailable.

### Adaptive Noise Scaling

The implementation adjusts uncertainty during extended dropout intervals.

As the period without real observations increases, the algorithm scales uncertainty to reduce overconfidence in predicted measurements.

### Method 1 Configuration

| Parameter | Value |
|-----------|-------|
| Initial height | 10 m |
| Launch speed | 20 m/s |
| Launch angle | 45° |
| Time step | 0.1 s |
| Simulation steps | 100 |
| Process noise covariance | diag(0.01, 0.01, 0.01, 0.01) |
| Measurement noise covariance | diag(0.5, 0.5) |
| Dropout start | Step 30 |
| Dropout end | Step 60 (exclusive) |
| Historical window | 5 |
| Polynomial degree | 2 |

### Method 1 Results

The notebook reports:

| Metric | Accuracy |
|--------|----------|
| Position accuracy | 92.94% |
| Velocity accuracy | 79.85% |

These values represent normalized estimation accuracy for the recorded simulation run.

## 🔄 Method 2: Kalman Filter with RTS Smoothing

The second approach combines standard Kalman filtering with the **Rauch–Tung–Striebel (RTS) smoother**.

### Concept

The approach performs two stages.

**Stage 1: Forward Kalman Filtering**

The filter:

1. Predicts the next state using the motion model.
2. Updates the estimate when measurements are available.
3. Continues prediction without measurement updates during dropouts.
4. Stores filtered and predicted states and covariance matrices.

**Stage 2: Backward RTS Smoothing**

After the forward filtering process, the algorithm performs a backward pass.

The backward pass uses information from later observations to refine earlier state estimates.

### RTS Smoothing Equations

The smoothing gain is:

\[
G_k=P_{k|k}A^T(P_{k+1|k})^{-1}
\]

The smoothed state is:

\[
\hat{x}_{k|N}
=
\hat{x}_{k|k}
+
G_k(\hat{x}_{k+1|N}-\hat{x}_{k+1|k})
\]

The smoothed covariance is:

\[
P_{k|N}
=
P_{k|k}
+
G_k(P_{k+1|N}-P_{k+1|k})G_k^T
\]

These equations allow estimates to be refined using observations from across the complete trajectory.

### Method 2 Configuration

| Parameter | Value |
|-----------|-------|
| Initial height | 10 m |
| Launch speed | 20 m/s |
| Launch angle | 45° |
| Time step | 0.1 s |
| Simulation steps | 100 |
| Process noise covariance | diag(0.01, 0.01, 0.01, 0.01) |
| Measurement noise covariance | diag(0.5, 0.5) |
| Dropout start | Step 30 |
| Dropout end | Step 80 (exclusive) |

### Method 2 Results

The notebook reports:

| Metric | Accuracy |
|--------|----------|
| Position accuracy (smoothed) | 98.17% |
| Velocity accuracy (smoothed) | 91.25% |

The saved results show accurate trajectory reconstruction in the simulated environment.

## 📊 Comparison of Both Methods

The following table summarizes the recorded results.

| Metric | Ensemble Pseudo-Measurements | RTS Smoothing |
|--------|------------------------------|---------------|
| Kalman filtering | Yes | Yes |
| Gaussian Process Regression | Yes | No |
| Polynomial Regression | Yes | No |
| Linear Regression | Yes | No |
| Pseudo-measurement generation | Yes | No |
| Backward smoothing | No | Yes |
| Dropout interval | Steps 30–59 | Steps 30–79 |
| Position accuracy | 92.94% | 98.17% |
| Velocity accuracy | 79.85% | 91.25% |

### Results Interpretation

The saved notebook results show higher position and velocity accuracy for the RTS smoothing approach.

However, the two simulations use different dropout durations.

Consequently, these results should be interpreted as separate experimental outcomes rather than a controlled benchmark establishing that RTS smoothing is universally superior.

A rigorous comparison would require identical settings, repeated random trials, and statistical analysis.

## 📏 Accuracy Evaluation

The notebook calculates normalized accuracy from Root Mean Squared Error (RMSE).

The general formula is:

\[
Accuracy(\%) =
100\left(1-\frac{RMSE}{Range}\right)
\]

Where:

- `RMSE` represents the estimation error.
- `Range` represents the relevant ground-truth value range.

Accuracy is calculated separately for position and velocity.

These percentages are custom normalized error metrics, not probabilities of correct predictions.

## 📈 Visualization

The project generates plots showing:

- True projectile trajectory
- Noisy sensor observations
- Estimated trajectory
- Smoothed trajectory in Method 2
- Differences between estimates and ground truth

The visualizations illustrate how the filters behave during normal measurements and extended dropout intervals.

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash id="x6i0rl"
git clone https://github.com/Fadinrsultan/kalman_filter_simulation.git

cd kalman_filter_simulation
```

### 2. Create a Virtual Environment

```bash id="oqegrq"
python -m venv .venv
```

Activate the environment.

**macOS / Linux:**

```bash id="x4wrz0"
source .venv/bin/activate
```

**Windows:**

```bash id="a78vla"
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash id="098y3g"
pip install numpy matplotlib scikit-learn notebook
```

### 4. Launch Jupyter Notebook

```bash id="hd6l7x"
jupyter notebook
```

Open:

```text id="fz0i6h"
kalman_filter_simulation.ipynb
```

Execute the notebook cells sequentially to run both experiments.

### 5. Run in Google Colab

Open:

https://colab.research.google.com/github/Fadinrsultan/kalman_filter_simulation/blob/main/kalman_filter_simulation.ipynb

## 🧪 Experimental Workflow

The simulation follows these general steps:

1. Initialize projectile motion parameters.
2. Generate the true trajectory.
3. Generate noisy sensor observations.
4. Introduce measurement dropouts.
5. Initialize the Kalman filter and state covariance.
6. Predict position and velocity.
7. Update using available observations.
8. Apply the selected dropout-handling strategy.
9. Estimate projectile states.
10. Calculate position and velocity RMSE.
11. Convert RMSE into normalized accuracy.
12. Visualize the results.

## ⚠️ Limitations

The project has several limitations:

- Projectile motion is simulated rather than measured by physical sensors.
- The motion model assumes constant gravitational acceleration.
- Air resistance is not included.
- Process and measurement noise are modeled using simplified covariance matrices.
- Measurement dropout intervals are predefined.
- Regression-based pseudo-measurements may become unreliable during long gaps.
- RTS smoothing requires a complete observation sequence and is not purely online.
- The two experiments use different dropout lengths.
- Reported accuracy results come from individual simulation runs.
- Random noise can influence the results.
- Real-world tracking systems would require further calibration and validation.

## 🔮 Future Improvements

Potential enhancements include:

- Extended Kalman Filter (EKF)
- Unscented Kalman Filter (UKF)
- Adaptive noise covariance estimation
- Automatic dropout detection
- Sensor fusion from multiple measurement sources
- Real-time target tracking
- Comparison with Particle Filters
- Improved pseudo-measurement uncertainty calibration
- Repeated Monte Carlo evaluation
- Confidence interval estimation
- Runtime and computational complexity analysis
- Integration with real sensor datasets
- Interactive trajectory visualization

## 🌐 Potential Applications

The methods explored in this project are relevant to:

- Robotics
- Autonomous navigation
- Radar tracking
- Aerospace trajectory estimation
- Computer vision
- Sensor fusion
- Motion prediction
- State estimation
- Tracking systems with intermittent observations

## 📚 References

### Kalman Filtering

**Kalman, R. E. (1960).**

*A New Approach to Linear Filtering and Prediction Problems.*

Journal of Basic Engineering.

https://doi.org/10.1115/1.3662552

### RTS Smoothing

**Rauch, H. E., Tung, F., & Striebel, C. T. (1965).**

*Maximum Likelihood Estimates of Linear Dynamic Systems.*

AIAA Journal.

https://doi.org/10.2514/3.3166

### Gaussian Process Regression

**Rasmussen, C. E., & Williams, C. K. I. (2006).**

*Gaussian Processes for Machine Learning.*

MIT Press.

https://gaussianprocess.org/gpml/

### Scientific Computing

- [NumPy Documentation](https://numpy.org/doc/)
- [Matplotlib Documentation](https://matplotlib.org/stable/)
- [Scikit-learn Gaussian Processes](https://scikit-learn.org/stable/modules/gaussian_process.html)

## 👨‍💻 Author

**Fadi Sultan**

AI & Machine Learning Developer

**GitHub Repository:**

https://github.com/Fadinrsultan/kalman_filter_simulation

---

⭐ If you find this project useful, consider giving the repository a star.
