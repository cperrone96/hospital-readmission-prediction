# Data provenance

- **Source:** UCI Machine Learning Repository, *Diabetes 130-US Hospitals for Years 1999-2008* (dataset 296).
- **Dataset citation:** Clore J, Cios K, DeShazo J, Strack B (2014). Diabetes 130-US Hospitals for Years 1999-2008 [Dataset]. UCI Machine Learning Repository. DOI [10.24432/C5230J](https://doi.org/10.24432/C5230J).
- **Paper:** Strack B, DeShazo JP, Gennings C, et al. Impact of HbA1c measurement on hospital readmission rates: analysis of 70,000 clinical database patient records. *BioMed Research International* 2014:781670.
- **License:** CC BY 4.0, as stated on the [UCI dataset page](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008).
- **Download:** `https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip`
  (SHA-256 of the zip as retrieved on 2026-10-23: `f82ac129da2ddd2299391ff6fbae3a6a58b3edcf59ac9d7bd480c00fe453112a`).

## Files in `data/`

| File | Contents | SHA-256 |
|---|---|---|
| `diabetic_data.csv.gz` | `diabetic_data.csv` from the zip, unmodified, compressed with `gzip -n -9` | uncompressed CSV: `0689e7ec031237dc63031b938805c48377748761a3b26acab621567afa24df97` |
| `IDS_mapping.csv` | Code tables for admission type, discharge disposition and admission source, unmodified | `f1bb82b471cb34649352597572c9b1fb00bd27f77b9f5a22a03dc3eb1039749e` |

The file has 101,766 encounters from 71,518 patients and keeps the source identifiers `encounter_id` and `patient_nbr`. `patient_nbr` is used only to split by patient and to cluster the bootstrap; neither identifier is a model feature.

To rebuild the data folder from the source:

```bash
curl -L -o uci296.zip "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"
unzip uci296.zip diabetic_data.csv IDS_mapping.csv
gzip -n -9 -c diabetic_data.csv > data/diabetic_data.csv.gz
mv IDS_mapping.csv data/
```

## History

Earlier versions of this repository shipped a re-export of the same file that dropped `encounter_id` and `patient_nbr` and blanked the `?` and `None` codes. Without patient IDs the split could only be by row, so the same patient could appear in training and test. The blanking also merged "test not performed" (`None`) with genuinely missing values. Both problems are fixed by using the source file above. The earlier DOI (10.24432/C53919) also pointed to a different UCI dataset (CDC Diabetes Health Indicators) and has been corrected.
