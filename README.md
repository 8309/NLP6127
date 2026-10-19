# Oil Evidence - AI6127

Portable retrieval demo, report, presentation, source code and evaluation results.

**Annotation provenance corrected on 19 October 2026.** The project owner confirms that the selected 200 DEVELOPMENT and 1,000 TEST records were human annotated and that the reported 1,200-pair agreement table measures two independent human annotators. Earlier descriptions of this selected cohort or its agreement as machine annotation were incorrect. Automatic answer generation/judging, older pilot records and machine source checks retain their own provenance. Reported label values and metric values are unchanged; the human streams were not independently re-audited in this correction. Historical filenames, schema keys and hashes are compatibility references, not the corrected annotation origin.

See [setup and demonstration instructions](README_TEACHER.md), [annotation correction](ANNOTATION_CORRECTION.md), [report](report.pdf), [presentation](presentation.pptx), and [human agreement values](results/human_agreement.json). The repository includes the 11,815-passage corpus, vector index, local embedding model and 1,000 TEST label records. The 200 DEVELOPMENT records and the two raw human annotation streams are not included. Historical generation logs are not bundled.

Run `SETUP.cmd`, then `RUN.cmd` on Windows. Model attribution is in [MODEL_NOTICES.md](MODEL_NOTICES.md); per-source metadata and redistribution conditions remain attached to the corpus.
