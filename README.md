 REINFORCE (Policy Gradient) on CartPole-v1
An implementation of **REINFORCE** (Monte Carlo policy gradient), written from scratch in PyTorch and trained on the classic **CartPole-v1** control task.

## Overview

REINFORCE is a policy-based deep reinforcement learning algorithm. It doesn't learn a value function. Instead, it directly optimizes a neural-network policy π(a|s) using gradient ascent on the expected return.

In CartPole, a pole is balanced on a cart that moves left or right along a track. The agent receives +1 reward for every timestep the pole stays upright, up to a maximum of 500 per episode.

## Files

| File | Description |
|---|---|
| `code.ipynb` | Executed Colab notebook with code, outputs, and answers |
| `reinforce_cartpole.pdf` | PDF export of the executed notebook |
| `cartpole_video.mp4` | Replay of the trained agent (optional) |

## Implementation

**Policy network**
- Input: 4 state values (cart position, cart velocity, pole angle, pole angular velocity)
- `fc1`: Linear(4 → 16) followed by ReLU
- `fc2`: Linear(16 → 2) followed by softmax, giving a probability for each action (push left or push right)
- Actions are **sampled** from the output distribution, which provides exploration and keeps the policy gradient estimate unbiased.

**Training loop (REINFORCE)**
1. Roll out one full episode with the current policy, saving the log-probabilities and rewards.
2. Compute the discounted return G_t for each timestep by iterating **backward** (G_t = r_t + γ·G_(t+1)), which takes O(N) time.
3. Normalize the returns (subtract the mean, divide by the standard deviation) for stable training.
4. Compute the loss as −Σ log π(a_t|s_t) · G_t, then take a gradient step with Adam.

## Hyperparameters

| Parameter | Value |
|---|---|
| Hidden size | 16 |
| Training episodes | 400 |
| Max steps per episode | 1000 |
| Discount factor (γ) | 1.0 |
| Learning rate | 1e-2 (Adam) |
| Evaluation episodes | 10 |

## Results

**Training**, as the average score over the last 100 episodes:

| Episode | Average score |
|---|---|
| 100 | 85.86 |
| 200 | 244.60 |
| 300 | 470.68 |
| 400 | 477.06 |

**Evaluation** (10 episodes): mean reward **456.2**, std **87.6**. The course's score is mean − std ≈ **368.6**, which is above the 350 target.

Because the policy is stochastic, evaluation results vary between runs. One run scored a perfect 500.0 ± 0.0.

## How to Run

1. Open the notebook in Google Colab.
2. Set **Runtime → Change runtime type → T4 GPU**. This is optional, since CartPole also trains quickly on CPU.
3. Run all cells in order.

### Environment notes
- Uses `gym==0.25.2`, the old Gym API in which `reset()` returns the state and `step()` returns `(state, reward, done, info)`.
- The original course requirements file pins `pyyaml==6.0`, which fails to build on Python 3.13. The dependencies are installed directly instead:
  ```bash
  pip install gym pygame imageio-ffmpeg
  pip install git+https://github.com/ntasfi/PyGame-Learning-Environment.git
  pip install git+https://github.com/simoninithomas/gym-games
  ```
- **NumPy 2 compatibility:** old Gym references `np.bool8`, which NumPy 2 removed. This is fixed by adding the alias before training:
  ```python
  np.bool8 = np.bool_
  ```

## References
- Hugging Face Deep RL Course, Unit 4: Policy Gradient with PyTorch
- Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.)
- PyTorch official REINFORCE example
