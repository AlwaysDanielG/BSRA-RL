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
  tool computes the motion you see in the viewer when a control is applied?

  MuJoCo does both. `mujoco.MjModel.from_xml_path` in
  [scripts/view_model.py](../scripts/view_model.py#L18) reads `scene.xml`, pulls
  `cartpole.xml` in at the `<include>`, and compiles the two files into one
  `MjModel`, which is the fixed description of the bodies, joints, geoms, and the
  motor. The changing state lives in a separate `MjData`. MuJoCo also computes the
  motion: the loop in [onboarding/viewer.py](../onboarding/viewer.py) writes the
  control into `data.ctrl` and calls `mujoco.mj_step`, which advances the physics
  by one `0.02 s` timestep, and the viewer just draws whatever that produced.

- Which XML file owns the mechanism, and how does the include connect it to the
  scene? Explain the slide/hinge axes, unactuated pole, box half-extents, and
  degrees versus radians. Link a small model screenshot (`../results/model.png`):

  [assets/cartpole.xml](../assets/cartpole.xml) owns the mechanism. It holds the
  `cart` and `pole` bodies and the `cart_motor` actuator inside a
  `<mujocoinclude>` root, so it is not a model on its own.
  [assets/scene.xml](../assets/scene.xml) is the top-level `<mujoco>` model, and
  its `<include file="cartpole.xml"/>` drops those contents in during compilation.
  The scene keeps the global settings and the scenery: compiler units, gravity,
  RK4, timestep, the floor, rail, light, and the `side` camera.

  The `slider` joint slides along `1 0 0`, so the cart moves in x, down the rail.
  The `hinge` rotates about `0 1 0`, so the pole falls in the x-z plane, the same
  plane the cart moves in. Only `slider` has a motor; the pole has no actuator, so
  it only moves from gravity and from being dragged by the cart. Box `size` values
  are half-extents, which is why `0.15 0.10 0.08` gives the `0.30 x 0.20 x 0.16 m`
  cart. The scene sets `angle="degree"`, so the hinge range is written as
  `-90 90` in the XML, but at runtime MuJoCo reports that joint in radians, and
  `qpos[1]` is limited to about `+/-1.5708`.

  ![Model screenshot](../results/model.png)

  The shot is the `side` camera after 25 steps at control `0.1`: the cart has slid
  along the rail and the pole has swung back about the hinge.

### Environment (Stage 3)

Answer each row in a few sentences, using the source linked in
[Stage 3](ONBOARDING.md#3-environment-setup-and-inspection). Name the function or
setting that supports your answer, and distinguish MuJoCo's role from Gymnasium's.

| Topic        | Question                                                                                                                                                | Your answer |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------- |
| Observations | What are the four values returned to the policy, in order and with units? Where do those values come from, and how does `_get_obs()` assemble them?     | Cart position (m), pole angle (rad), cart velocity (m/s), pole angular velocity (rad/s). They come straight from MuJoCo's simulation data: `_get_obs()` does `np.concatenate([self.data.qpos, self.data.qvel]).ravel()`, and my XML has two joints, so `qpos` and `qvel` are two elements each. The order follows the XML, `slider` first and `hinge` second. Gymnasium only copies the numbers out; MuJoCo is what computed them. |
| Actions      | What does the action control, what is its allowed range, and how does it reach the motor?                                                               | One value, the force pushing the cart along the rail. The range is `[-3, 3]`, which Gymnasium reads off the `ctrlrange` I set on `cart_motor` in [assets/cartpole.xml](../assets/cartpole.xml), not from anything in the Python. It reaches the motor in `_step_mujoco_simulation()`, which does `self.data.ctrl[:] = ctrl`. The `gear="100"` on the motor then scales it, so an action of 3 is not 3 N at the joint. |
| Physics      | Which tool computes motion from your XML model, and how does Gymnasium ask it to advance? How much simulated time passes per action?                    | MuJoCo computes the motion. `step()` calls `do_simulation(action, self.frame_skip)`, which passes it to `_step_mujoco_simulation()`, which calls `mujoco.mj_step(..., nstep=n_frames)`. That is the same `mj_step` the viewer used in Stage 2. With `timestep=0.02` from my scene and `FRAME_SKIP = 2`, each action covers `0.04 s` of simulated time (`dt = timestep * frame_skip`), so the policy acts at 25 Hz. |
| Reset        | What does `reset_model()` change at the start of an episode? How are the starting values randomized, and what does using the same reset seed reproduce? | It resets all four state values. It takes `init_qpos` and `init_qvel`, which are the zero/upright state from my XML, adds uniform noise in `[-0.01, 0.01]` (`RESET_NOISE` passed in as `reset_noise_scale`), and calls `set_state()`. The noise comes from `self.np_random`, so the same reset seed gives the exact same starting state every time, which is what makes the PPO-versus-random comparison in Stage 5 fair. |
| Reward       | What is the exact reward rule, which code computes it, and does it use the state before or after the action?                                            | `reward = int(not terminated)` in `step()`, so 1 for every step survived and 0 on the step that fails. It uses the state *after* the action: `step()` simulates first, then calls `_get_obs()`, then checks `terminated`. Failure is `abs(observation[1]) > 0.2` rad of pole angle, or a non-finite observation. Cart position never fails, and my XML's `[-1.25, 1.25]` slider limit only blocks the cart, it does not end the episode. |

- Environment test and random-rollout results; action-range warning:

  `uv run python -m pytest test/test_env.py` passes all 11 tests. Random rollouts
  (`scripts/evaluate.py --random-only --run-name random-debug --episodes 3`) gave a
  mean return of 4.67 and mean length 5.67 steps, with every episode ending on
  `terminated` and none hitting the time limit, so random pushing drops the pole in
  well under half a second. The `--video` run wrote
  `videos/random-debug/episode-10000.mp4` at 640 x 480, 25 fps, and my custom cart,
  pole, rail, and floor are visible through the `side` camera.

  SB3's env checker warns: `UserWarning: We recommend you to use a symmetric and
  normalized Box action space (range=[-1, 1])`. I am keeping `[-3, 3]` because the
  onboarding says to, and because the range comes from the motor's `ctrlrange` in
  my XML.

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
