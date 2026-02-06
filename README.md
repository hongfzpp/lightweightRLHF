# JAX RLHF Learning Pipeline

A hands-on, exercise-driven RLHF (Reinforcement Learning from Human Feedback) pipeline built entirely in JAX. Designed as a learning project for post-training research engineering roles.

## What You'll Build

A complete RLHF pipeline covering all three stages:

```
Stage 1: SFT (Supervised Fine-Tuning)
    |
    v
Stage 2: Reward Model Training
    |
    +---> Stage 3A: PPO  (Proximal Policy Optimization)
    +---> Stage 3B: DPO  (Direct Preference Optimization)
    +---> Stage 3C: GRPO (Group Relative Policy Optimization)
```

The project uses a small GPT-2 model (~10M params) that trains in minutes on a single GPU, so you can focus on understanding the algorithms rather than waiting for training.

## Target Platform

**Apple Silicon M4** (Mac Studio Mini) with the Metal GPU backend.

- Uses `jax-metal` for GPU acceleration via Metal
- `float16` mixed precision (Metal does not support `bfloat16`)
- Multi-device exercises use CPU simulation (`XLA_FLAGS`)
- Unified memory = no CPU-GPU transfer overhead

The code also works on CUDA GPUs and TPUs with minor changes (replace `jax-metal` with `jax[cuda12]`).

## Setup

```bash
# 1. Create a virtual environment (Python 3.11 recommended for jax-metal)
python3.11 -m venv .venv
source .venv/bin/activate

# 2. Install JAX with Metal backend
pip install jax jax-metal

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Verify installation
python -c "import jax; print(f'Backend: {jax.default_backend()}'); print(f'Devices: {jax.devices()}')"
# Expected: Backend: metal (or gpu)
```

## How It Works

The project is structured as **27 exercises** across 8 phases, ordered from easy to hard. Each exercise:

1. Has a clear **docstring** explaining what the code achieves
2. Explains **why JAX** is a good fit compared to PyTorch/NumPy
3. Provides **step-by-step hints** in the comments
4. Contains a `raise NotImplementedError("EXERCISE X.Y")` placeholder for you to fill in
5. Has **unit tests** to verify your implementation

### Workflow

```bash
# 1. Read the exercise description in the source file
# 2. Implement the function (follow the hints in the comments)
# 3. Run the tests to verify
pytest tests/test_jax_basics.py -v    # Phase 1

# 4. Once all tests pass, move to the next exercise
# 5. After completing a phase, run the corresponding script
python scripts/run_sft.py             # After Phase 3
```

## Exercise Index

### Phase 1: JAX Fundamentals (Easy) -- `utils/jax_basics.py`

| Exercise | Function | JAX Concept | Est. Time |
|----------|----------|-------------|-----------|
| 1.1 | `linear()`, `init_linear_params()` | `jax.numpy` arrays, pure functions | 20 min |
| 1.2 | `mse_loss()`, `compute_gradients()` | `jax.grad`, `jax.value_and_grad` | 20 min |
| 1.3 | `train_step()` | `jax.jit` (XLA compilation) | 30 min |
| 1.4 | `per_example_loss()`, `batched_loss()` | `jax.vmap` (auto-vectorization) | 30 min |

**Test:** `pytest tests/test_jax_basics.py -v`

### Phase 2: Model Architecture (Easy-Medium) -- `models/`

| Exercise | File | What You Build | Est. Time |
|----------|------|----------------|-----------|
| 2.1 | `attention.py` | Multi-head causal self-attention | 60 min |
| 2.2 | `transformer_block.py` | Transformer block (attn + FFN + residual) | 30 min |
| 2.3 | `gpt2.py` | Full GPT-2 LM with weight tying | 60 min |

**Test:** `pytest tests/test_models.py -v`

### Phase 3: SFT Training (Medium) -- `algorithms/sft.py`

| Exercise | Function | What You Learn | Est. Time |
|----------|----------|----------------|-----------|
| 3.1 | `cross_entropy_loss()` | Next-token prediction loss with label masking | 30 min |
| 3.2 | `sft_train_step()` | JIT-compiled train step with Optax | 45 min |
| 3.3 | `sft_eval_step()` | Evaluation without gradients | 15 min |

**Test:** `pytest tests/test_sft.py -v` | **Run:** `python scripts/run_sft.py`

### Phase 4: Reward Model (Medium) -- `models/reward_model.py` + `algorithms/reward.py`

| Exercise | Function | What You Learn | Est. Time |
|----------|----------|----------------|-----------|
| 4.1 | `RewardModel.__call__()` | Scalar reward head on GPT-2 backbone | 45 min |
| 4.2 | `preference_loss()` | Bradley-Terry preference loss | 30 min |
| 4.3 | `reward_train_step()` | Training with `has_aux=True` | 30 min |

**Test:** `pytest tests/test_reward.py -v` | **Run:** `python scripts/run_reward.py`

### Phase 5: PPO (Hard) -- `algorithms/ppo.py`

| Exercise | Function | What You Learn | Est. Time |
|----------|----------|----------------|-----------|
| 5.1 | `compute_kl_divergence()` | KL penalty, `jax.lax.stop_gradient` | 30 min |
| 5.2 | `compute_gae()` | GAE with `jax.lax.scan` | 60 min |
| 5.3 | `ppo_policy_loss()` | PPO clipped surrogate objective | 45 min |
| 5.4 | `value_function_loss()` | Clipped value loss | 30 min |
| 5.5 | `ppo_update_step()` | Combined PPO update | 60 min |

**Test:** `pytest tests/test_ppo.py -v` | **Run:** `python scripts/run_ppo.py`

### Phase 6: DPO (Medium-Hard) -- `algorithms/dpo.py`

| Exercise | Function | What You Learn | Est. Time |
|----------|----------|----------------|-----------|
| 6.1 | `compute_response_log_probs()` | Sequence log-probs for preference pairs | 30 min |
| 6.2 | `dpo_loss()` | DPO loss with implicit reward | 45 min |
| 6.3 | `dpo_train_step()` | 4-forward-pass training step | 30 min |

**Test:** `pytest tests/test_dpo.py -v` | **Run:** `python scripts/run_dpo.py`

### Phase 7: GRPO (Hard) -- `algorithms/grpo.py`

| Exercise | Function | What You Learn | Est. Time |
|----------|----------|----------------|-----------|
| 7.1 | `group_sample_and_score()` | Group sampling with `jax.random.split` | 45 min |
| 7.2 | `group_relative_advantage()` | Z-score normalization (no value network) | 20 min |
| 7.3 | `grpo_loss()` | PPO-clip with group advantages + KL | 45 min |

**Test:** `pytest tests/test_grpo.py -v` | **Run:** `python scripts/run_grpo.py`

### Phase 8: Advanced (Expert) -- `advanced/`

| Exercise | File | What You Learn | Est. Time |
|----------|------|----------------|-----------|
| 8.1 | `multi_device.py` | `jax.sharding`, `NamedSharding`, `Mesh` | 90 min |
| 8.2 | `mixed_precision.py` | float16 forward / float32 grads, loss scaling | 60 min |
| 8.3 | `iterative_dpo.py` | On-policy preference generation, ref model updates | 90 min |

**Test:** `pytest tests/test_advanced.py -v`

### Full Pipeline

After completing Phases 2-5, run the end-to-end pipeline:

```bash
python scripts/run_full_pipeline.py
```

This runs SFT -> Reward Model -> PPO in sequence.

## Project Structure

```
JAX_rlhf/
├── README.md                     # This file
├── requirements.txt              # Dependencies (JAX, Flax, Optax, etc.)
│
├── configs/                      # Hyperparameter configurations
│   ├── model_config.py           # GPT-2 architecture (TINY/SMALL/MEDIUM)
│   ├── sft_config.py             # SFT training params
│   ├── reward_config.py          # Reward model training params
│   ├── ppo_config.py             # PPO params (clip_eps, gae_lambda, etc.)
│   ├── dpo_config.py             # DPO params (beta, label_smoothing)
│   └── grpo_config.py            # GRPO params (group_size, kl_coeff)
│
├── data/                         # Data loading (provided, not exercises)
│   ├── tokenizer.py              # GPT-2 tokenizer wrapper
│   ├── sft_dataset.py            # (prompt, completion) pairs
│   └── preference_dataset.py     # (prompt, chosen, rejected) triples
│
├── models/                       # Model definitions
│   ├── attention.py              # EXERCISE 2.1
│   ├── transformer_block.py      # EXERCISE 2.2
│   ├── gpt2.py                   # EXERCISE 2.3
│   ├── reward_model.py           # EXERCISE 4.1
│   └── policy.py                 # Log-prob/entropy utilities (provided)
│
├── algorithms/                   # Core RLHF algorithms
│   ├── sft.py                    # EXERCISES 3.1-3.3
│   ├── reward.py                 # EXERCISES 4.2-4.3
│   ├── ppo.py                    # EXERCISES 5.1-5.5
│   ├── dpo.py                    # EXERCISES 6.1-6.3
│   └── grpo.py                   # EXERCISES 7.1-7.3
│
├── utils/                        # Utilities (provided, not exercises)
│   ├── jax_basics.py             # EXERCISES 1.1-1.4
│   ├── jax_utils.py              # RNG, params, masking helpers
│   ├── checkpointing.py          # Orbax save/load
│   ├── logging_utils.py          # MetricsTracker
│   └── generation.py             # Top-k / nucleus sampling
│
├── training/                     # Training orchestration (provided)
│   ├── sft_trainer.py            # SFT training loop
│   ├── reward_trainer.py         # Reward model loop
│   ├── ppo_trainer.py            # PPO loop (rollout + update)
│   ├── dpo_trainer.py            # DPO loop
│   └── grpo_trainer.py           # GRPO loop
│
├── advanced/                     # Expert-level exercises
│   ├── multi_device.py           # EXERCISE 8.1
│   ├── mixed_precision.py        # EXERCISE 8.2
│   └── iterative_dpo.py          # EXERCISE 8.3
│
├── scripts/                      # Run scripts
│   ├── run_sft.py
│   ├── run_reward.py
│   ├── run_ppo.py
│   ├── run_dpo.py
│   ├── run_grpo.py
│   └── run_full_pipeline.py
│
└── tests/                        # Unit tests (one per phase)
    ├── test_jax_basics.py
    ├── test_models.py
    ├── test_sft.py
    ├── test_reward.py
    ├── test_ppo.py
    ├── test_dpo.py
    ├── test_grpo.py
    └── test_advanced.py
```

## Learning Path

### Recommended Order

1. **Phase 1** (1-2 hours): Get comfortable with JAX transforms
2. **Phase 2** (2-3 hours): Build the model architecture in Flax
3. **Phase 3** (1-2 hours): Train the model with SFT
4. **Phase 4** (1-2 hours): Train a reward model
5. **Phase 5** (4-6 hours): Implement PPO (the core of RLHF)
6. **Phase 6** (2-3 hours): Implement DPO (simpler alternative)
7. **Phase 7** (3-4 hours): Implement GRPO (modern approach)
8. **Phase 8** (4-6 hours): Advanced topics

Total estimated time: **~20-30 hours**

### Tips

- **Run tests early and often.** Each exercise has specific tests. Don't move on until tests pass.
- **Read the comments carefully.** The step-by-step hints in each exercise file are designed to guide you through the implementation.
- **Start with the simple version.** For GAE (Exercise 5.2), implement the Python loop first, then try the `jax.lax.scan` version.
- **Use the TINY_CONFIG for debugging.** It's faster and uses less memory.
- **Don't modify infrastructure files** (data/, training/, utils/ except jax_basics.py). They're designed to work once you implement the exercises.

### JAX Concepts You'll Master

| Concept | Where Used | Why It Matters for RLHF |
|---------|-----------|------------------------|
| `jax.numpy` | Phase 1 | Foundation for all computation |
| `jax.grad` | Phase 1, 3-7 | Differentiating complex loss functions |
| `jax.jit` | All phases | Compiling training loops for Metal GPU |
| `jax.vmap` | Phase 1, 6, 7 | Batching over preference pairs / groups |
| `jax.lax.scan` | Phase 5 | Compiled loops for GAE |
| `jax.lax.stop_gradient` | Phase 5, 6 | Freezing reference model in loss |
| `jax.random` | Phase 1, 7 | Explicit PRNG for reproducible sampling |
| `jax.checkpoint` | Phase 2 | Memory-efficient training (rematerialization) |
| `jax.sharding` | Phase 8 | Multi-device / distributed training |
| Flax `nn.Module` | Phase 2, 4 | Model definition with init/apply pattern |
| Optax | Phase 3-7 | Composable optimizers (AdamW + clipping) |
| Pytree operations | All phases | Manipulating parameter trees |

## Key Algorithms

### PPO (Phase 5)
The classic RLHF approach (InstructGPT/ChatGPT). Generates responses, scores with reward model, computes advantages via GAE, updates with clipped objective.

### DPO (Phase 6)
Simpler alternative: no reward model needed. Directly optimizes policy from preference pairs using an implicit reward formulation.

### GRPO (Phase 7)
DeepSeek's approach: eliminates the value network by sampling a group of responses per prompt and normalizing rewards within each group.

## Apple M4 Notes

- **Backend verification:** Every script prints the JAX backend on startup. Look for `metal` or `gpu`.
- **No bfloat16:** Metal does not support bfloat16. All mixed-precision code uses float16.
- **Unified memory:** The M4's unified memory means no CPU-GPU transfer overhead, which is an advantage for PPO (constant switching between generation and training).
- **Memory:** With the SMALL_CONFIG (~10M params), the full PPO setup (policy + reference + value head + reward model) fits comfortably in 16GB+.

## License

This is a learning project. Use it freely for study and portfolio purposes.
