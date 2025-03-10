# SMBOX: Sequential Model-Based Optimization eXpress Algorithm

## Process Diagram

```
+-----------------------------------------------------------------------------------+
|                            SMBOX HPO Algorithm Flow                               |
+-----------------------------------------------------------------------------------+

┌────────────────────────┐
│  Input Configuration   │
│  - Algorithm type      │
│  - Parameter space     │
│  - Objective function  │
│  - Time budget         │
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│  Initialization Phase  │
│  1. Create low-fidelity│
│     dataset (optional) │
│  2. Generate initial   │
│     random population  │
│  3. Evaluate initial   │
│     configurations     │
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│  Create History Table  │
│  - Store evaluations   │
│  - Track global best   │
│  - Initialize gen = 0  │
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│  Update Config Schema  │
│  - Adapt param bounds  │
│  - Update distributions│
│  - Focus on promising  │
│    regions             │
└───────────┬────────────┘
            │
            ▼
            ┌───────────────────────────────────────┐
            │              Main Loop                │
            └─────────────────┬─────────────────────┘
                              │
                              ▼
┌────────────────────────┐    │    ┌────────────────────────┐
│  Fit Response Surface  │◄───┘───►│  Generate Candidates   │
│  - Train CatBoost model│         │  - Create large pool   │
│  - Learn from history  │         │  - Apply feasibility   │
│  - Predict new configs │         │    checks              │
└───────────┬────────────┘         └───────────┬────────────┘
            │                                   │
            ▼                                   ▼
┌────────────────────────┐         ┌────────────────────────┐
│  Predict Performance   │         │  Selection Strategy    │
│  - Score candidates    │         │  - Select top configs  │
│  - Rank by predicted   │         │  - Add exploration     │
│    performance         │         │    configs (random)    │
└───────────┬────────────┘         │  - Add pseudo-random   │
            │                       │    configs (optional) │
            │                       └───────────┬────────────┘
            └───────────────────►              │
                                               ▼
                                 ┌────────────────────────┐
                                 │  Evaluate Population   │
                                 │  - Run objective func  │
                                 │  - Score configurations│
                                 │  - Track performance   │
                                 └───────────┬────────────┘
                                             │
                                             ▼
                                 ┌────────────────────────┐
                                 │  Update History        │
                                 │  - Add new evaluations │
                                 │  - Update global best  │
                                 │  - Track improvements  │
                                 └───────────┬────────────┘
                                             │
                                             ▼
                                 ┌────────────────────────┐
              ┌─────────────────►│  Time Budget Check     │
              │                  │  - Continue iteration  │
              │                  │    if time remains     │
              │                  │  - Exit if time up     │
              │                  └───────────┬────────────┘
              │                              │
              │                              ▼
              │                  ┌────────────────────────┐
              └──────────No─────┤  Termination Check     │
                                 │  Is time_status == 'OK'│
                                 └───────────┬────────────┘
                                             │
                                             │ Yes
                                             ▼
                                 ┌────────────────────────┐
                                 │  Return Best Result    │
                                 │  - Best parameters     │
                                 │  - Best performance    │
                                 │  - Save trials data    │
                                 └────────────────────────┘
```

## Key Components and Features

1. **Initialization Phase**:
   - Creates a low-fidelity dataset for faster initial evaluation (if configured)
   - Generates an initial population of random hyperparameter configurations
   - Evaluates these initial configurations to build a foundation for learning

2. **Response Surface Model**:
   - Uses CatBoost as a surrogate model to learn the relationship between hyperparameters and performance
   - Trained on history of previously evaluated configurations

3. **Adaptive Parameter Space**:
   - Updates the parameter search space based on promising regions
   - Focuses exploration on areas with high-performing configurations

4. **Candidate Generation & Selection**:
   - Generates a large pool of potential hyperparameter configurations
   - Predicts their performance using the surrogate model
   - Selects the most promising candidates while maintaining exploration

5. **Exploration Strategies**:
   - Injects random configurations to prevent premature convergence
   - Optional pseudo-random configurations for targeted exploration
   - Balance between exploitation (using best known areas) and exploration (trying new areas)

6. **Time-Based Termination**:
   - Runs until a predefined time budget is exhausted
   - Returns the best configuration found within the time constraints

7. **Performance Tracking**:
   - Logs all evaluations and tracks global best
   - Monitors improvements throughout the optimization process
   - Integrates with MLflow for experiment tracking (when enabled)

## Algorithm Details

The SMBOX (Sequential Model-Based Optimization eXpress) algorithm efficiently navigates the hyperparameter space by learning from previous evaluations, focusing on promising regions, and maintaining a balance between exploration and exploitation. This approach is particularly effective for expensive black-box optimization problems like hyperparameter tuning.

### Meta-Parameters

The algorithm is controlled by several meta-parameters that determine its behavior:

- `lf_init_ratio`: Ratio for low-fidelity dataset creation during initialization
- `lf_init_n`: Number of initial random configurations to evaluate
- `lf_ratio`: Ratio for low-fidelity dataset during main optimization
- `alpha_n`: Number of top candidates to select in each iteration
- `inc_rand`: Whether to include random configurations (Y/N)
- `inc_pseudo_rand`: Whether to include pseudo-random configurations (Y/N)

### Benefits of SMBOX

1. **Efficiency**: Optimizes expensive hyperparameter configurations with fewer evaluations
2. **Adaptability**: Adjusts search space based on promising areas
3. **Balance**: Maintains effective exploration-exploitation trade-off
4. **Time-awareness**: Works within specified time constraints
5. **Flexibility**: Can be used with any machine learning algorithm
6. **Scalability**: Works with both simple and complex parameter spaces 