# AGENTS.md

## Project

`fs-cone-kit` is a small, command-line-first training pipeline for Formula Student Driverless cone detection. It wraps Ultralytics YOLO, uses Hydra for configuration, prepares FSOCO or an existing YOLO dataset, logs training results, and exports ONNX models.

Keep the project small and useful to teams outside Linköping University. Prefer a focused change over adding a new abstraction, backend, or workflow.

## Start here

Before changing code:

1. Read `README.md` for the supported workflow and user-facing promises.
2. Trace the relevant Hydra config from `configs/config.yaml` into its config group and `_target_` implementation.
3. Inspect the base class beside an implementation before changing its interface.

The main entrypoint is:

```bash
uv run -m core.train
```

Use `uv` for the environment and commands. Keep runtime options in Hydra configs when they are choices a user may reasonably override. Keep secrets out of configs and source files; document them in `.env.example` with empty values.

## Repository map

- `core/train.py`: orchestration and training entrypoint
- `core/data/`: dataset download, preparation, and adapters
- `core/trainers/`: trainer backends
- `core/loggers/`: experiment logger integrations
- `core/metrics/`: metric extraction and reporting
- `configs/`: Hydra composition and component settings
- `docs/`: README media
- `outputs/`, `data/`, `wandb/`, `runs/`: generated or local artifacts; do not commit them

## Implementation rules

- Preserve the split between orchestration, dataset adapters, trainers, loggers, and metrics. Put source-specific conversion in a dataset adapter rather than `core/train.py`.
- Configure components through Hydra `_target_` entries and instantiate them at the existing composition boundary.
- Treat the class map as one contract shared by labels, `dataset.yaml`, metrics, plots, and logger output. A class change is complete only when every one of those consumers agrees.
- Use `pathlib.Path` for filesystem paths. Do not assume the command runs from a particular working directory when a path can be anchored to the repository or the current Hydra run.
- Keep generated datasets, downloaded weights, credentials, logs, and training outputs out of version control.
- Avoid notebooks and manual UI-only setup. The documented path should remain reproducible from commands and config.
- Add a dependency only when the standard library and existing dependencies do not solve the problem cleanly. Update `uv.lock` together with `pyproject.toml`.

## Formula Student Germany rules

Formula Student Germany (FSG) is the main competition and its rules are the authority for competition-dependent behavior. The official, year-aware Rules & Documents page is [fsg.one/rules](https://fsg.one/rules), which redirects to the FSG website. It contains the Formula Student Rules, the Driverless Specification, the FSG Event Handbook, changelogs, and related documents; some material may require an FSG account.

For work affected by competition rules:

1. Open the official Rules & Documents page at the time of the change.
2. Use the documents for the relevant competition year and check their version and changelog.
3. Read the general Formula Student Rules together with the Driverless Specification and FSG Event Handbook where applicable.
4. Record the year, document version, and rule or section identifier in the issue, PR, code comment, or documentation that depends on it.

Do not treat a remembered rule, an old local PDF, another competition's rules, or a search-result excerpt as authoritative. Link to the official page instead of committing FSG documents to this repository.

## README voice

Treat the current `README.md` as the style source of truth. Before adding or rewriting README text, read the whole current file and analyze its voice, sentence length, point of view, heading style, amount of detail, and use of examples. Then make the addition sound as though the same author wrote it in the same editing pass.

At present, that voice is:

- first-person when explaining motivation, and direct second-person when guiding the reader;
- terse, practical, and confident, with short sentences and short paragraphs;
- plain-spoken and mildly blunt rather than promotional or corporate;
- command-first: show the exact command or config, then explain only what the reader needs;
- honest about limitations and defaults, without inflated claims;
- lowercase sentence-style headings and list items, except for proper names;
- occasional dry emphasis such as “The command line is the UI.” or “The repo is not magic.”, used sparingly.

Preserve the centered logo and demo markup unless the task explicitly changes the presentation. Keep examples copy-pastable and keep paths, defaults, and artifact locations synchronized with the code. Do not append generic boilerplate sections merely because they are conventional; every README addition must help someone install, configure, run, inspect, or extend this project.

## Verification

Match verification to the change:

- For Python changes, run the narrowest relevant check, then run `uv run -m core.train` when the change can be exercised by the documented debug job.
- For config changes, compose or start the affected Hydra path and confirm overrides still resolve.
- For dataset changes, use debug mode or a small fixture and inspect the generated `dataset.yaml`, image/label pairing, and class IDs.
- For README changes, execute or otherwise verify every changed command and confirm every named path and default against the repository.

Training can download data or weights and can initialize external logging. Use debug-sized runs, avoid expensive full training unless it is explicitly required, and state clearly when a check was skipped because credentials, network access, hardware, or runtime cost made it impractical.

Before handing off, inspect the diff, remove unrelated changes, and report what was verified and what was not.
