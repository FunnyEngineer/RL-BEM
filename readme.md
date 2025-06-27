# Project Plan: ML-Enhanced EnergyPlus with RL (Time Series Focus)

This plan outlines the steps to create a machine learning (ML) surrogate model for EnergyPlus simulations, enhanced with reinforcement learning (RL) for active learning, using **time series data** (e.g., hourly indoor temperature, energy use) within a 10-day timeframe. Each phase includes detailed tasks, estimated times, and checkpoints to monitor progress.

---

## Phase 2: Time Series Dataset Preparation (Days 3-4)
**Objective**: Process the ResStock time series dataset to extract relevant features and targets for model training.  
**Total Estimated Time**: 14 hours

- [x] **Task 2.1**: Select relevant features and targets  
  - **Action**: Choose input features from both metadata (e.g., building type, vintage, insulation) and time series (e.g., outdoor temperature, hour of day). Select time series targets such as hourly indoor temperature (`out.zone_mean_air_temp.conditioned_space.c`) and/or energy use.  
  - **Details**: Merge metadata with time series for each building. Confirm feature and target availability in both datasets.  
  - *Estimated time*: 2 hours  

- [x] **Task 2.2**: Clean and preprocess the time series data  
  - **Action**: Load the time series dataset using `pandas` (from Parquet files). Handle missing timestamps, align time indices, and resample if needed. Impute missing values (e.g., forward fill for time series, median/mode for static). Normalize numerical features (e.g., insulation R-value, temperature) using Min-Max scaling or standardization. Encode categorical variables (e.g., building type) with one-hot encoding or label encoding. Encode time features (hour, day, season) as cyclical variables.  
  - **Details**: Check for outliers (e.g., spikes in energy use) and decide whether to cap or remove them.  
  - *Estimated time*: 6 hours  

- [x] **Task 2.3**: Split the data  
  - **Action**: Divide the dataset into a training pool (80%), validation set (10%), and test set (10%). Split by **building** to avoid leakage, and within each, split time series into train/val/test periods or use cross-validation over time. Use sliding windows for training (e.g., predict next 24h from previous 168h).  
  - **Details**: Ensure splitting is computationally efficient (e.g., use `train_test_split` for buildings, and windowing for time).  
  - *Estimated time*: 2 hours  

- [x] **Task 2.4**: Prepare data for efficient access  
  - **Action**: Save the preprocessed training pool, validation, and test sets as Parquet files for fast loading in later phases. Use multi-index DataFrames (building, timestamp) if needed. Optionally, create a smaller sample (e.g., 10K time steps) for quick testing.  
  - **Details**: Use `pandas.to_parquet()` to handle large datasets efficiently.  
  - *Estimated time*: 4 hours  

**Checkpoint**: By the end of Day 4, you'll have a preprocessed ResStock time series dataset split into training, validation, and test sets, ready for sequence model training.

---

## Phase 3: Initial Surrogate Sequence Model (Day 5)
**Objective**: Train an ML sequence model to approximate EnergyPlus time series outputs.  
**Total Estimated Time**: 8 hours

- [x] **Task 3.1**: Preprocess the dataset for sequence modeling  
  - Normalize inputs and outputs.  
  - Create sliding windows (e.g., use past 168h to predict next 24h).  
  - *Estimated time*: 2 hours  

- [x] **Task 3.2**: Train a sequence model  
  - Build a sequence model (RNN, LSTM, GRU, TCN, or transformer) using `pytorch`.  
  - Predict time series targets (e.g., indoor temperature, energy use) from time series and static features.  
  - *Estimated time*: 4 hours  

- [x] **Task 3.3**: Evaluate the model  
  - Compute validation metrics (e.g., MAE, RMSE, DTW) on time series.  
  - Plot predicted vs. actual time series curves.  
  - *Estimated time*: 2 hours  

**Checkpoint**: By end of Day 5, you should have a functional surrogate sequence model with reasonable accuracy.

---

## Phase 4: RL for Active Learning with Time Series (Days 6-8)
**Objective**: Implement an RL agent to select optimal simulation points or control actions over time.  
**Total Estimated Time**: 18 hours

- [x] **Task 4.1**: Design the RL environment  
  - **State**: Recent time series (model error, recent weather, etc.).  
  - **Action**: Select next input parameters (static or time-varying, e.g., thermostat setpoints, schedules).  
  - **Reward**: Reduction in time series prediction error, or achieving temporal goals (e.g., comfort, peak reduction).  
  - *Estimated time*: 4 hours  

- [x] **Task 4.2**: Implement a simple RL algorithm  
  - Use Q-learning or policy gradient with a discretized action space (e.g., setpoint schedules).  
  - Set hyperparameters (learning rate, discount factor, exploration rate).  
  - *Estimated time*: 6 hours  

- [x] **Task 4.3**: Integrate RL with EnergyPlus and surrogate sequence model  
  - Update simulation script to use RL-selected time series inputs.  
  - Retrain the surrogate model with new time series data.  
  - *Estimated time*: 8 hours  

- [x] **Task 4.4**: Integrate RL with surrogate sequence model  
  - ✅ Created `ActiveLearningLoop` class for iterative RL-driven data collection
  - ✅ Implemented RL agent training and new sample collection
  - ✅ Added surrogate model retraining with expanded dataset
  - ✅ Created `run_active_learning.py` script for easy execution
  - ✅ Added comprehensive logging and evaluation metrics
  - *Estimated time*: 8 hours

**Checkpoint**: By end of Day 8, you should have an RL agent selecting new simulation points or control actions and updating the dataset.

---

## Phase 5: Iterative Improvement (Days 9-10)
**Objective**: Iteratively improve the surrogate sequence model using RL.  
**Total Estimated Time**: 10 hours

- [x] **Task 5.1**: Run the RL loop  
  - Execute 10-20 iterations: select point/control, run simulation, retrain model.  
  - Monitor validation error improvement on time series.  
  - *Estimated time*: 10 hours (run overnight if needed)  

- [x] **Task 5.2**: Optimize simulation time  
  - Reduce iterations or use a faster surrogate if simulations are slow.  
  - *Estimated time*: Ongoing  

**Checkpoint**: By morning of Day 10, you should have an improved surrogate sequence model.

---

## Evaluation and Wrap-Up (Day 10)
**Objective**: Assess and document the final model.  
**Total Estimated Time**: 7 hours

- [x] **Task 6.1**: Compare final and initial models  
  - Evaluate both on a test set or via cross-validation (time series metrics).  
  - Measure accuracy improvement.  
  - *Estimated time*: 2 hours  

- [ ] **Task 6.2**: Compare to a baseline  
  - Random agent baseline script is being added for direct comparison with RL-guided models.
  - *Estimated time*: 3 hours  

- [ ] **Task 6.3**: Document findings  
  - Documentation and summary script will be added after running all experiments.
  - *Estimated time*: 2 hours  

**Final Deliverable**: An RL-enhanced surrogate sequence model with documented performance.

---

## Running All DQN Agents and Baselines on LS6

To run all DQN agent variants and the random agent baseline on LS6:

1. **Submit the batch job:**
   - Use the provided `submit_rl.sh` script (see below for the updated version) to run all agents in sequence. This will execute `run_all_dqn_agents.sh` and the random agent baseline script, saving all results in the `reports/` directory.

2. **Check results:**
   - Learning curves, reward trajectories, and agent checkpoints will be saved in `reports/`.

3. **Compare and document:**
   - Use the provided comparison and documentation scripts (to be added) to analyze and summarize results.

---

## Progress Tracking Suggestions
- **Checklist**: Use this markdown file as a checklist—mark tasks with `[x]` as completed.  
- **Trello Board**: Create columns for "To Do," "In Progress," and "Done." Add tasks as cards.  
- **Daily Goals**: Set targets (e.g., finish Phase 1 by Day 2) and review nightly.  
- **Time Management**: Allocate hours per task and use a timer.  
- **Version Control**: Use Git to track code and data changes.

---

## Additional Tips
- **Simulation Speed**: Run simulations overnight or use a cloud service if slow.  
- **Simplify**: Reduce simulation count or use variance-based sampling if time is short.  
- **Debugging**: Reserve time for integration issues (e.g., RL-EnergyPlus linkage).  
- **Resources**:  
  - [EnergyPlus Python API](https://energyplus.readthedocs.io/en/latest/api.html)  
  - [Active Learning Guide](https://towardsdatascience.com/active-learning-in-machine-learning-1c3754b8f7e0)  
  - [Time Series Forecasting with Deep Learning](https://pytorch-forecasting.readthedocs.io/en/stable/)  
  - [Stable Baselines3 RL](https://stable-baselines3.readthedocs.io/en/master/)  

Good luck! Reach out if you need help with any step.

---

## 2025-06-27 – Codebase Refactor for Flexible Surrogate Models

A lightweight, **architecture-agnostic** training workflow was introduced to
replace the monolithic `train_surrogate_model.py` script.

Key points

* `src/modeling/models/` – contains all network architectures.  New models are
  registered in `models/__init__.py` and selected at runtime via
  `--model <key>`.
* `BaseSequenceModel` – common LightningModule with boiler-plate training logic
  (optimiser, LR-scheduler, logging) to avoid code duplication.
* `LSTMSurrogateModel` – re-implemented to inherit from the base class and live
  in `models/lstm_model.py`.
* `src/modeling/train_sequence_model.py` – **new** generic training entry-point
  that accepts arbitrary architectures plus any extra hyper-parameters:
  
  ```bash
  # Train default LSTM
  python -m src.modeling.train_sequence_model --version my_run
  
  # Train GRU once implemented
  python -m src.modeling.train_sequence_model --model gru --hidden_size 256 \
         --version gru_run
  ```
* `TimeSeriesDataset` & `create_data_loaders()` moved to
  `src/modeling/datasets.py` for reuse.

### Deprecation Notice

`src/modeling/train_surrogate_model.py` is **deprecated**.  It is still kept for
legacy scripts (e.g. `active_learning_loop.py`) but will be removed in a future
release. A `DeprecationWarning` is emitted on import to encourage migration.

Scripts that only needed the surrogate model class have been updated to use the
new path:

```
from src.modeling.models.lstm_model import LSTMSurrogateModel
```

Please migrate custom notebooks and pipelines accordingly.

### Import style change (Lightning ≥ 2)

All source files now follow the official ≥ v2 syntax:

```python
import lightning as L

class MyModel(L.LightningModule):
    ...

trainer = L.Trainer(...)
```

The legacy alias `import pytorch_lightning as pl` has been removed from the
codebase to avoid confusion.  If you work in notebooks remember to update your
imports as well.