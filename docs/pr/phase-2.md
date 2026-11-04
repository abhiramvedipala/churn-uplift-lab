## Phase 2: uplift modeling on a simulated retention offer

### What was built
- `churnlab/simulate.py`: seeded 50/50 offer on top of out-of-fold churn probabilities, with persuadables, sleeping dogs (two-year contracts) and lost causes (fiber + electronic check). True per-customer uplift is saved.
- `churnlab/uplift.py`: T-learner (two XGBoost models) and causalml uplift random forest, Qini/AUUC on observed outcomes, true uplift captured in the top 20%, Qini chart and uplift SHAP.

### How it was tested
- 5 tests: simulation contract (seed, columns, effect signs by segment) and uplift ranking beating random.

### Numbers (held-out n=2,113)
| Targeting | Qini | Share of oracle, top 20% |
|---|---|---|
| Uplift forest | 0.875 | 91.9% |
| T-learner | 0.761 | 81.8% |
| Churn risk | - | 45.2% |
| Random | - | 26.6% |

### Open questions
- Simulation v1 (no lost causes) let churn-risk targeting reach 94.7% of the oracle, so uplift modeling added nothing. v2 adds lost causes; both are in history (`3c00724`, `6a70b06`). Real data has to show which world we are in.
