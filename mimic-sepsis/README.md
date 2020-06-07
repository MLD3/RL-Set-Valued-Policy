# Experiments on MIMIC-sepsis task

## Step 1: Extract data (`AI_Clinician-forked`)
- This is a subset of code in https://github.com/matthieukomorowski/AI_Clinician that deals with data extraction. 
- Software versions: 
    - postgres/psql (PostgreSQL) 9.5.19
    - Python 3.7.4
    - MATLAB R2016b
- Requires a built MIMIC-III database (and concepts) in PostgreSQL.
- Only run steps I, II, and III. 
    - Run notebook `AIClinician_Data_extract_MIMIC3_140219.ipynb` to extract data from the database and saves output in `exportdir/`: < 1hr
    - `run('AIClinician_sepsis3_def_160219.m')` identifies the cohort and saves it `sepsis_mimiciii.csv`: ~2hr
    - `run('AIClinician_MIMIC3_dataset_160219.m')` produces the dataset `MIMICtable.csv`: ~2hr
- Both matlab scripts have been modified for errors and bug fixes. 
- After data extraction finishes running, copy the extracted files `sepsis_mimiciii.csv` and `MIMIC_dataset.csv` to `mimic_sepsis_rl/data/`.

## Step 2: Main experiments (`mimic_sepsis_rl`)
- Software versions:
    - Python 3.7.4
- Required pip packages:
    - pandas, numpy, scipy, sklearn
    - matplotlib, seaborn
    - joblib, tqdm
- Follow instructions in `1_preprocess` to create `trajD_tr.pkl`, `trajD_va.pkl`, `trajD_te.pkl` which contain the trajectories split into train/val/test sets
- Copy these files into `2_learn`, and run the notebooks in numerical order
