# Oil Evidence - AI6127

Portable retrieval demo, report, presentation, source code and evaluation results.

**Annotation provenance corrected on 19 October 2026.** The delivery contains 1,000 TEST records with human reference annotations. Earlier machine-origin descriptions of these labels were incorrect. Automatic answer generation and judging, older pilot records and machine source checks retain their own provenance. Questions, label values and TEST evaluation scores are unchanged. Overall raw human annotation agreement is 91.13% for TEST. Per-field agreement and kappa values are not reported. Historical filenames, schema keys and hashes are compatibility references.

See [setup and demonstration instructions](README_TEACHER.md), [annotation correction](ANNOTATION_CORRECTION.md), [report](report.pdf), [presentation](presentation.pptx), and [TEST agreement](results/human_agreement.json). The repository includes the 11,815-passage corpus, vector index, local embedding model and 1,000 TEST label records. The two raw human annotation streams are not included. Historical generation logs are not bundled.

Run `SETUP.cmd`, then `RUN.cmd` on Windows. Model attribution is in [MODEL_NOTICES.md](MODEL_NOTICES.md); per-source metadata and redistribution conditions remain attached to the corpus.
