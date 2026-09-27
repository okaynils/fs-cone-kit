# experimental framework plan

This milestone keeps Ultralytics, Hydra, and the existing dataset/trainer/logger
boundaries. It adds four small pieces around them:

1. Make FSOCO splits deterministic and persist a manifest with every prepared
   dataset and experiment.
2. Write a self-contained experiment record: resolved and redacted config,
   source/dependency metadata, dataset identity, training metrics, and weights.
   Allow a saved config or `last.pt` to start the same run again.
3. Add standalone final-test evaluation and inference benchmarking commands.
   Store stable JSON records, including per-class metrics, confusion matrices,
   model size, timing protocol, and hardware.
4. Compare completed experiment records locally and export CSV and Markdown.
   Refuse to imply that results from different splits or benchmark setups are
   directly comparable.

The automated tests exercise metadata redaction, result serialization,
comparison rules, and persistent split identity. A separate CPU-only smoke
command runs the real training, evaluation, timing, and report path on generated
images. Full FSOCO training and GPU timing remain manual checks because they
require the dataset, model weights, and suitable hardware.
