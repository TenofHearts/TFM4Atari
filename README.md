# TFM4Atari

TFM4Atari uses TabPFN 3.5 as an Atari policy. Pretrained RL Zoo teachers supply
cold-start trajectories and executed actions. One TabPFN classifier learns from
two complete teacher trajectories in a single combined original context. The
configured policy uses rolling symbolic outcome conditioning and probability
sampling; frozen behavior cloning remains available as the comparison mode.

The default configuration enables **BeamRider only**. The other six benchmark
games are defined but are not collected or tested until added to
`enabled_games` in `config.toml`.

## Setup

Python dependencies and cross-platform resolutions are pinned by `uv.lock`:

```powershell
uv --cache-dir .\.uv-cache sync --group dev
uv run tfm4atari preflight
```

TabPFN 3.5 is license-gated. Complete the normal Prior Labs authentication once
on each machine before `bootstrap-judges`. Authentication tokens are not
project configuration and are never written to `config.toml`.

The default checkpoint is pinned to
`tabpfn-v3.5-20260909.safetensors`. Its first CPU load can take several minutes;
the configured cache lives under `artifacts/tabpfn` and can be copied to a new
server together with the project data.

## Pipeline

All operational settings come from `./config.toml`; commands intentionally have
no settings flags.

```powershell
uv run tfm4atari preflight
uv run tfm4atari fetch-teachers
uv run tfm4atari prepare-tabpfn
uv run tfm4atari collect
uv run tfm4atari bootstrap-judges
uv run tfm4atari play
uv run tfm4atari evaluate
uv run tfm4atari record-teacher-video
uv run tfm4atari record-video
uv run tfm4atari record-learning-video
```

`tfm4atari pipeline` runs the same stages in order. Teacher episodes and
learning trials are immutable Parquet parts, so interrupted collection and play
commands resume from completed work.

## Learning contract

- The DQN sees RL Zoo-compatible stacked 84×84 grayscale frames.
- TabPFN sees 128 RAM bytes, signed one-step RAM deltas, and compact action/game
  state fields. It never receives pixels.
- Exactly two naturally completed teacher episodes form the cold start.
  Incomplete episodes and episodes assigned any other collection role cannot
  enter this context. Every executed action from both teacher episodes is kept;
  the game-specific online relevance filter is never applied to teacher data.
- Teacher loading is backend-pluggable through `config.toml`: DQN and QR-DQN
  expose action values, while PPO exposes action probabilities as preferences.
  Stored cold-start trajectories and RL queues are isolated by backend.
- Every teacher action receives a rolling label from its overlapping forward
  16-action window. The existing symbolic interval rule is reused unchanged:
  `+1` if the window contains reward, `-1` if it contains death/punishment, and
  `0` otherwise, with death/punishment taking precedence. Teacher
  Q-values/probabilities are not used as context labels.
- Online experience, when enabled, retains completed 8-action cache batches and
  uses the same reward/death/neutral symbolic rule.
- `learning.policy_mode = "behavior_cloning"` omits the symbolic label during
  fit and inference. `"outcome_conditioned"` includes the rolling label during
  fit and requests `learning.desired_symbolic_label` during inference.
- `learning.action_selection` selects deterministic greedy actions, samples
  directly from predicted action probabilities, or applies configured epsilon
  sampling. Every stochastic mode is reset from the reproducible episode seed.
- Online adaptation is enabled for the configured multi-episode learning-video
  experiment. Set `context_cache.enabled = false` for frozen teacher-context
  comparisons where self-generated actions must not affect later decisions.
- Adaptive online PFN episodes never call the teacher. The game-specific relevance
  filter decides which executed actions enter each 8-action batch by default.
  BeamRider keeps every action because the player is continuously active. The
  game-specific interval labeler labels the completed batch with one flag.
  Every batch is persisted immediately; on CPU the actor is rebuilt after 32
  judged batches (256 decisions) to avoid conflating durable experience capture
  with the much more expensive TabPFN refit cadence.
- `data/<game>/context_cache_<backend>_<label-schema>_<policy-mode>_`
  `<selection-mode>.parquet` is strictly a
  bounded FIFO queue
  of those judged, actually executed actions. It is not a search tree: cached
  states are never revisited, branched, or used to resample actions.
- When the cache is enabled, the actor is refit during an episode on both
  complete cold-start trajectories and the FIFO queue. Context budgeting
  reserves explicit space for recent additions while retaining at least the
  configured number of teacher rows. Behavior-cloning teacher downsampling is
  proportional across trajectory and action strata; outcome-conditioned
  sampling additionally preserves symbolic-label strata.
- When the cache is enabled, `record-video` can persist its filtered judged
  actions, making the recorded episode a learning addition rather than a
  read-only demonstration. This is controlled in `config.toml`.
- `record-learning-video` runs `video.learning_episodes` sequential adaptive
  episodes. It carries the bounded rolling-label cache across episodes, persists
  every episode as a learning addition, checkpoints per-episode scores and
  diagnostics to JSON, and losslessly joins the episode recordings into one
  MP4. `video.retain_learning_episode_videos` controls whether the component
  clips remain after the joined video is finalized. This command requires both
  `context_cache.enabled` and `video.persist_context_additions` to be true.
- Game-specific judges and relevance filters live under
  `src/tfm4atari/games/<game>/`. BeamRider uses a symbolic progress/survival
  judge and an all-actions relevance filter.

The pretrained teacher was trained on legacy `NoFrameskip-v4`. This project uses
the current `ALE/<Game>-v5` registration with deterministic base frames and the
legacy RL Zoo preprocessing protocol. `preflight` and evaluation must pass
before results are trusted.
