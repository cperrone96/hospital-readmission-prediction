# Data provenance

- Source: UCI Machine Learning Repository, *Diabetes 130-US Hospitals for Years 1999-2008*.
- Citation: Strack et al. (2014), DOI [10.24432/C53919](https://doi.org/10.24432/C53919).
- Retrieval/source page: https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008
- Checked-in file: `data/diabetic_data.csv.gz` (the repository's legacy export; it has 48 columns and omits `encounter_id` and `patient_nbr`).
- SHA-256: `1d2d901d5998f39ee73242d9cd259d01ae12d1a7978a6614d21d60ad8434fca9`.

The analysis checks for the source identifiers and refuses to label a row-level split patient-disjoint when they are unavailable. No identifiers are recreated.
