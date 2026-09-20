# TFM4Atari

TFM4Atari uses TabPFN 3.5 as a context-conditioned Atari policy. A pretrained
RL Zoo DQN supplies cold-start trajectories and action Q-values. One TabPFN
classifier learns `action_success` from two complete teacher trajectories in a
single combined original context, then extends that context with judged actions
from its own gameplay.

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
  enter this context.
- Teacher loading is backend-pluggable through `config.toml`: DQN and QR-DQN
  expose action values, while PPO exposes action probabilities as preferences.
  Stored cold-start trajectories and RL queues are isolated by backend.
- At every teacher state, actions in the tied upper Q-value quartile receive
  `action_success=true`. Raw Q-values and normalized advantages are retained.
- Online PFN episodes never call the teacher. Every 64 decisions by default,
  the game-specific relevance filter first discards irrelevant executed
  actions. The pluggable trajectory judge then labels the retained actions
  with the completed window's success flag.
- `data/<game>/context_cache_<backend>.parquet` is strictly a bounded FIFO queue
  of those judged, actually executed actions. It is not a search tree: cached
  states are never revisited, branched, or used to resample actions.
- The actor is refit during an episode on both complete cold-start trajectories
  plus the FIFO queue. Context budgeting reserves explicit space for the newest
  RL-like additions while retaining at least the configured number of teacher
  rows; additions cannot disappear during teacher-context downsampling.
- By default, `record-video` also persists its filtered judged actions, making
  the recorded agent episode a real RL-like learning addition rather than a
  read-only demonstration. This is controlled in `config.toml`.
- Game-specific judges and relevance filters live under
  `src/tfm4atari/games/<game>/`. BeamRider uses a symbolic progress/survival
  judge and a combat/reward/life-transition relevance filter.

The pretrained teacher was trained on legacy `NoFrameskip-v4`. This project uses
the current `ALE/<Game>-v5` registration with deterministic base frames and the
legacy RL Zoo preprocessing protocol. `preflight` and evaluation must pass
before results are trusted.
