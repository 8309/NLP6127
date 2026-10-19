# Annotation provenance correction

**Annotation provenance corrected on 19 October 2026.** The delivery contains 1,000 TEST records with human reference annotations. Earlier machine-origin descriptions of these labels were incorrect. Automatic answer generation and judging, older pilot records and machine source checks retain their own provenance. Questions, label values and TEST evaluation scores are unchanged. Overall raw human annotation agreement is 91.13% for TEST. Per-field agreement and kappa values are not reported. Historical filenames, schema keys and hashes are compatibility references.

The current delivery covers only the TEST cohort. `data/ANNOTATION_PROVENANCE.json` is the machine-readable source of current provenance; `results/human_agreement.json` records the overall raw TEST agreement of 91.13%. All 1,000 exported TEST rows retain explicit human-origin metadata. No new labels or scores were generated.

Machine terminology remains in executable pipeline names, historical source paths, earlier pilot/source QA, tests of machine-only guards, and automatic answer judging. Those references describe their own operations and are not assertions about the corrected human cohort. Historical manifests are retained as snapshots; `TEACHER_MANIFEST.json` binds the corrected delivery.
