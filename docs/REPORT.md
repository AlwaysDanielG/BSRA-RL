# CartPole report

Replace each **TODO** with your own observations. Keep this report short; link code,
compact results, and images instead of pasting logs.
For each stage, explain which tool or function uses your inputs, where the
results come from, and what uses them next. Point to the relevant code or
documentation; a few sentences or an annotated arrow diagram is enough.

To link a file in Markdown, use `[link text](relative/path/to/file)`.
For example, `[Environment code](../onboarding/env.py)` becomes
[Environment code](../onboarding/env.py). Links are relative to this report's
`docs/` folder; use `../` to reach the repository root.
To display an image, add `!`: `![Model screenshot](../results/model.png)`.

## Setup

- Member: Daniel Gong
- OS: 15.6
- Setup diagnostic, viewer, and RGB results; any fix needed: Finished
- Fork URL and working branch: https://github.com/grmpn/BSRA-RL.git  https://github.com/AlwaysDanielG/BSRA-RL.git
  Before starting, read the [toolchain overview](../resources/toolchain.md).
  What role does each of MuJoCo, Gymnasium, Stable-Baselines3, and TensorBoard
  play in this exercise? Describe how they work together in your own words: 

  MuJoCo is the physics engine. It loads the model files and figures out how the cart and pole actually move when a force is applied to the cart. Gymnasium sits on top of that and turns the simulation into a normal RL task, so it decides what the agent sees, what actions it can take, what reward it gets, and when an episode ends. Stable-Baselines3 is what does the learning. It talks to the Gymnasium environment, runs episodes over and over, and trains a PPO policy for me so I don't have to write the algorithm myself. TensorBoard just shows the numbers that SB3 logs while training, like reward over time, so I can see if things are improving and compare different runs.

  This is how they fit: SB3 picks an action, Gymnasium passes it into MuJoCo, MuJoCo steps the physics forward, and the new state comes back as an observation and a reward. That loop repeats thousands of times, and TensorBoard lets me look at what happened.

## Model and task

### Model (Stage 2)

- Which tool loads `scene.xml` and its included `cartpole.xml`, and what does it
  create from them? Point to the loading call in `scripts/view_model.py`. Which
  tool computes the motion you see in the viewer when a control is applied? **TODO**
- Which XML file owns the mechanism, and how does the include connect it to the
  scene? Explain the slide/hinge axes, unactuated pole, box half-extents, and
  degrees versus radians. Link a small model screenshot (`../results/model.png`): **TODO**

### Environment (Stage 3)

Answer each row in a few sentences, using the source linked in
[Stage 3](ONBOARDING.md#3-environment-setup-and-inspection). Name the function or
setting that supports your answer, and distinguish MuJoCo's role from Gymnasium's.

| Topic        | Question                                                                                                                                                | Your answer |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------- |
| Observations | What are the four values returned to the policy, in order and with units? Where do those values come from, and how does `_get_obs()` assemble them?     | **TODO**    |
| Actions      | What does the action control, what is its allowed range, and how does it reach the motor?                                                               | **TODO**    |
| Physics      | Which tool computes motion from your XML model, and how does Gymnasium ask it to advance? How much simulated time passes per action?                    | **TODO**    |
| Reset        | What does `reset_model()` change at the start of an episode? How are the starting values randomized, and what does using the same reset seed reproduce? | **TODO**    |
| Reward       | What is the exact reward rule, which code computes it, and does it use the state before or after the action?                                            | **TODO**    |

- Environment test and random-rollout results; action-range warning: **TODO**

## Training

- Run name / source commit / training seed: **TODO**
- Requested and actual steps / elapsed time / device: **TODO**
- Configuration and versions: **TODO** link `../results/<run>/run.json`
- Learning curve: **TODO** embed `../results/<run>/learning-curve.png`

- What do SB3, PyTorch, and `Monitor` each do here? How does SB3 use the environment
  and settings you pass to PPO? **TODO**
- What do the curve's axes, averaging, and smoothing mean, and what evidence of
  improvement, variability, or a plateau do you see? **TODO**

## Evaluation

Use reset seeds 10000–10019, deterministic PPO actions, and random action seeds
`reset seed + 20000`. Standard deviations describe episodes (`ddof=0`).

| Agent | Episodes | Return mean ± std | Length mean ± std | Time-limit fraction |
| --- | --- | --- | --- | --- |
| Random | TODO | TODO | TODO | TODO |
| PPO | TODO | TODO | TODO | TODO |

- Per-episode CSV and summary JSON: **TODO** links
- Predetermined PPO rollout (seed 10000): **TODO** briefly describe the behavior you observed
- Limitation or failure, and a specific diagnosis if learning was weak: **TODO**

- Why does this evaluation differ from a training curve, and what can one
  trained seed not establish? What carries over to humanoid soccer, and what is
  missing (for example, contacts, partial observations, or sim-to-real transfer)? **TODO**

## Review

**TODO:** Record exact setup, test, train, and evaluate commands, plus any rendering
environment variable.

Keep the saved policy, TensorBoard logs, and rollout video locally and share them
directly with the RL lead if requested. No file uploads, download links, or GitHub
release are required for these files.

- Automated test results: **TODO**
- Manual model/render/video checks: **TODO**
- Fork PR link and review notes (record merge in the PR): **TODO**

## Feedback

- Did you learn anything from this onboarding? What was new, or what became
  clearer? If little was new, say so: **TODO**
- What improvements would make the onboarding easier to follow or more useful?
  Mention any confusing instructions, missing background, or unnecessary work: **TODO**
