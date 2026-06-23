import pandas as pd
import numpy as np
import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer


from config import CONFIG

# Preprocessing for the NephroCAGE dataset v1

files = {
        'baseline_parameters': '1_BAseline_parameter_fertig2_HLA_Pirche_final.xlsx',
        'donorparameters': '2_donoparameter_final.xlsx',
        'exams': '4_exams.csv',
        'biopsy': '5 Biopsy_patho_kreuz_extensive.xlsx',
        'lab_cohort': '6 Lab_cohort.csv',
        'clinical_assessment': '7_clinical_assessment.csv',
        'medication': '8_Medikation.csv',
        'hospitalization': '9_Hospitalization.xlsx',
        'hla': '10_HLA-DSA_timecourse.xlsx'
    }

def read_files(project_path):
    # reads files returns the dataframe for each file
    data_folder = 'data/v1'
    dfs = {}

    # date of transplant, gender, date of birth, dialysis... of 3878 patients (> num patients in hosp)
    dfs['baseline_parameters'] = pd.read_excel(os.path.join(project_path, data_folder, files['baseline_parameters']))

    dfs['clinical_assessment'] = pd.read_csv(os.path.join(project_path, data_folder, files['clinical_assessment']), sep=';', encoding='ISO-8859-1', low_memory=False)

    dfs['exams'] = pd.read_csv(os.path.join(project_path, data_folder, files['exams']), sep=';', encoding='ISO-8859-1')

    # lab measurements of e.g. Kreatinin, Protein
    dfs['lab_cohort'] = pd.read_csv(os.path.join(project_path, data_folder, files['lab_cohort']), sep=';', encoding='ISO-8859-1')

    dfs['biopsy'] = pd.read_excel(os.path.join(project_path, data_folder, files['biopsy']))

    #dfs['hla'] = pd.read_excel(os.path.join(project_path, data_folder, files['hla']))

    # date of death, medication, date of transplant
    dfs['medikation'] = pd.read_csv(os.path.join(project_path, data_folder, files['medication']), sep=';', encoding='ISO-8859-1')

    # hosp start and end
    dfs['hospitalization'] = pd.read_excel(os.path.join(project_path, data_folder, files['hospitalization']), engine='openpyxl')

    # info about donor type, bloodgroup, age etc.
    dfs['donorparameters'] = pd.read_excel(os.path.join(project_path, data_folder, files['donorparameters']), engine='openpyxl')

    return dfs

def get_dfs(project_path):
    dfs = read_files(project_path)
    return dfs


def create_static_df(dfs):
    # expects dfs read from the original files
    # removes NaNs patient_ids, donor_ids, transplantation_ids
    # see preprocessing notebook
    # returns static df, matching patients to donors
    static_df = dfs['baseline_parameters'][['PatientID', 'TransplantationID', 'SpenderID', 'Geburtsdatum', 'Todesdatum', 'Datum', 'Geschlecht', 'Grunderkrankung', 'Datum_erste_Dialyse', 'Dialyse_Anzahl', 'Alter', 'Blutgruppe', 'Koerpergroesse', 'Date of graft loss', 'Loss cause', 'cold ischemia time', 'PIRCHE_Score', 'MMA_broad', 'MMB_broad', 'MMDR_broad', 'MM_broad']].rename(
        columns={
            'PatientID': 'patient_id',
            'SpenderID': 'donor_id',
            'TransplantationID': 'transplant_id',
            'Geburtsdatum': 'birth_date',
            'Todesdatum': 'death_date',
            'Datum': 'transplant_date',
            'Geschlecht': 'gender',
            'Grunderkrankung': 'underlying_disease',
            'Datum_erste_Dialyse': 'first_dialysis_date',
            'Dialyse_Anzahl': 'number_dialyses',
            'Alter': 'age',
            'Blutgruppe': 'blood_group',
            'Koerpergroesse': 'height',
            'Date of graft loss': 'loss_date',
            'Loss cause': 'loss_cause',
            'cold ischemia time': 'cold_ischemia_time',
            'PIRCHE_Score': 'pirche_score',
            'MMA_broad': 'mma_broad',
            'MMB_broad': 'mmb_broad',
            'MMDR_broad': 'mmdr_broad',
            'MM_broad': 'mm_broad', # sum of above three HLA mismatches, A,B,DR are different loci 
        }
    )

    static_df = static_df.dropna(subset=['patient_id', 'donor_id', 'transplant_id'])
    static_df = static_df.merge(
        dfs['donorparameters'][['PatientID', 'TransplantationID', 'SpenderID', 'gender_donor', 'age_donor', 'donor_bloodgroup',
                                'type_of_donation', 'donor_COD', 'donor_weight', 'donor_height', 'SOURCE']],
        how='left',  # Left join to keep all records in static_df
        left_on=['patient_id', 'transplant_id', 'donor_id'],
        right_on=['PatientID', 'TransplantationID', 'SpenderID']
    )

    static_df = static_df.drop(columns=['PatientID', 'TransplantationID', 'SpenderID'])
    static_df = static_df.drop_duplicates(subset='patient_id', keep='first')

    static_df['transplant_date'] = pd.to_datetime(static_df['transplant_date'], dayfirst=True, errors='coerce')
    static_df['loss_date'] = pd.to_datetime(static_df['loss_date'], dayfirst=True, errors='coerce')
    static_df['death_date'] = pd.to_datetime(static_df['death_date'], dayfirst=True, errors='coerce')

    # if loss date is given calculate the relative days from transplantation date
    static_df['loss_rel_days'] = (static_df['loss_date'] - static_df['transplant_date']).dt.days
    # same for death date
    static_df['death_rel_days'] = (static_df['death_date'] - static_df['transplant_date']).dt.days

    # selecting relevant features
    # these are further filtered by the actual features in CONFIG.py
    static_df = static_df[
        [
            'patient_id',
            'transplant_id',  # keep for medication merging
            'transplant_date',  # keep for medication merging
            'gender',
            'age',
            'underlying_disease',
            'number_dialyses',
            'blood_group',
            'height',
            'death_rel_days',
            'loss_rel_days',
            'loss_cause',
            'cold_ischemia_time',
            'gender_donor',
            'age_donor',
            'donor_bloodgroup',
            'type_of_donation',
            'donor_height',
            'donor_weight',
            'pirche_score',
            'mma_broad',
            'mmb_broad',
            'mmdr_broad',
            'mm_broad',
        ]
    ]

    mm_cols = ['mma_broad', 'mmb_broad', 'mmdr_broad', 'mm_broad']

    for col in mm_cols:
        static_df[col] = pd.to_numeric(static_df[col], errors='coerce').astype('Int64')  # Keeps NaNs and ensures integer dtype

    static_df['gender'] = static_df['gender'].str.lower()
    static_df['gender_donor'] = static_df['gender_donor'].str.lower()
    static_df['gender_donor'] = static_df['gender_donor'].apply(lambda x: x if x in ['m', 'w'] else 'm')

    static_df['type_of_donation'] = static_df['type_of_donation'].apply(lambda x: np.nan if pd.isna(x) else 'hirntot' if 'hirntot' in str(x).lower() else 'lebend')

    static_df['pirche_score'] = static_df['pirche_score'].astype(str).str.replace('_x000D_', '', regex=False) # remove artifacts
    static_df['pirche_score'] = pd.to_numeric(static_df['pirche_score'], errors='coerce').round(4)

    # Process blood groups
    static_df['blood_group'] = static_df['blood_group'].str.strip()
    static_df['donor_bloodgroup'] = static_df['donor_bloodgroup'].str.strip()
    
    # Convert standalone blood groups to include +
    static_df.loc[static_df['blood_group'].isin(['A', 'B', 'AB', '0']), 'blood_group'] += '+'
    static_df.loc[static_df['donor_bloodgroup'].isin(['A', 'B', 'AB', '0']), 'donor_bloodgroup'] += '+'
    
    # Set invalid blood groups to NaN
    valid_blood_groups = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', '0+', '0-']
    static_df.loc[~static_df['blood_group'].isin(valid_blood_groups), 'blood_group'] = np.nan
    static_df.loc[~static_df['donor_bloodgroup'].isin(valid_blood_groups), 'donor_bloodgroup'] = np.nan

    return static_df


def create_vitals_df(dfs):
    # expects raw dfs and returns two dfs (vitals from clinical assessments, vitals from lab cohort)

    static_df = create_static_df(dfs)

    vitals_ca = dfs['clinical_assessment'][['PatientID', 'TransplantationID', 'OPDtime', 'Blutdruck_systolisch', 'Blutdruck_diastolisch', 'Gewicht', 'Urinvolumen', 'Herzfrequenz', 'Temperatur', 'Diuresezeit']].rename(
        columns={
            'PatientID': 'patient_id',
            'TransplantationID': 'transplant_id',
            'OPDtime': 'rel_days', # date of assessment in days after transplantation
            'Blutdruck_systolisch': 'bp_sys',
            'Blutdruck_diastolisch': 'bp_dia',
            'Gewicht': 'weight',
            'Urinvolumen': 'urine_volume',  # lower urine volume indicates issues with kidney function
            'Herzfrequenz': 'hr',
            'Temperatur': 'temperature',
            'Diuresezeit': 'diuresis_time',  # duration over which urine output is measured
        }
    )

    # List of features to convert to numeric
    features_to_clean = ['bp_sys', 'bp_dia', 'weight', 'urine_volume', 'hr', 'temperature', 'diuresis_time']

    # Replace commas with dots and convert to numeric for the specified features
    for feature in features_to_clean:
        vitals_ca[feature] = vitals_ca[feature].astype(str).str.replace(',', '.')
        vitals_ca[feature] = pd.to_numeric(vitals_ca[feature], errors='coerce')


    print(f"Unique patients in clinical assessments: {vitals_ca['patient_id'].nunique()}")
    print('Removing patients that are not in static_df')
    vitals_ca = vitals_ca.merge(
        static_df[['patient_id', 'transplant_id']],
        how='inner',  # Inner join to keep only matching entries
        on=['patient_id', 'transplant_id']
    )

    print(f"Unique patients in clinical assessments: {vitals_ca['patient_id'].nunique()}")
    print(f"Average entries per patient {len(vitals_ca) / vitals_ca['patient_id'].nunique()}")


    vitals_lab = dfs['lab_cohort'][['PatientID', 'TransplantationID', 'Labtime', 'Bezeichnung', 'Wert', 'Einheit']].rename(
        columns={
            'PatientID': 'patient_id',
            'TransplantationID': 'transplant_id',
            'Labtime': 'rel_days', # date of assessment in days after transplantation
            'Bezeichnung': 'description',
            'Wert' : 'value',
            'Einheit': 'unit',
        }
    )

    print(f"Unique patients in lab df: {vitals_lab['patient_id'].nunique()}")
    print('Removing patients that are not in static_df')
    vitals_lab = vitals_lab.merge(
        static_df[['patient_id', 'transplant_id']],
        how='inner',  # Inner join to keep only matching entries
        on=['patient_id', 'transplant_id']
    )

    print(f"Unique patients in lab df: {vitals_lab['patient_id'].nunique()}")
    print(f"Average entries per patient {len(vitals_lab) / vitals_lab['patient_id'].nunique()}")

    vitals_ca['rel_days'] = vitals_ca['rel_days'].astype(str).str.replace(',', '.').astype(float).astype(int)
    vitals_lab['rel_days'] = vitals_lab['rel_days'].astype(str).str.replace(',', '.').astype(float).astype(int)

    return vitals_ca, vitals_lab


def create_medication_df(dfs):
    # expects raw dfs, returns medication df

    static_df = create_static_df(dfs)

    medication = dfs['medikation'][['PatientID', 'TransplantationID', 'prescription start', 'prescription end', 'Bezeichnung', 'DDD', 'unit', 'ATC']].rename(
        columns={
            'PatientID': 'patient_id',
            'TransplantationID': 'transplant_id',
            'prescription start': 'p_start',
            'prescription end': 'p_end',
            'Bezeichnung': 'description',
            'DDD': 'ddd', # defined daily dose
            'unit': 'unit',
            'ATC': 'atc' # the ATC column identifies the medication based on its pharmacological classification.
        }
    )

    print(f"Unique patients in medication: {medication['patient_id'].nunique()}")
    print('Removing patients that are not in static_df')
    medication = medication.merge(
        static_df[['patient_id', 'transplant_id', 'transplant_date']],
        how='inner',  # Inner join to keep only matching entries
        on=['patient_id', 'transplant_id']
    )

    # Calculate the relative days from transplantation date to prescription start and end
    medication['start'] = (pd.to_datetime(medication['p_start'], dayfirst=True)-
                                            pd.to_datetime(medication['transplant_date'], dayfirst=True)
                                            ).dt.days
    medication['end'] = (pd.to_datetime(medication['p_end'], dayfirst=True)-
                                            pd.to_datetime(medication['transplant_date'], dayfirst=True)
                                            ).dt.days
    medication = medication.drop(columns=['p_start', 'p_end', 'transplant_date'])
    print(f"Unique patients in clinical assessments: {medication['patient_id'].nunique()}")
    print(f"Average entries per patient {len(medication) / medication['patient_id'].nunique():.1f}")

    medication = medication.dropna(subset=['start', 'end'])
    medication['start'] = medication['start'].astype(str).str.replace(',', '.').astype(float).astype(int)
    medication['end'] = medication['end'].astype(str).str.replace(',', '.').astype(float).astype(int)

    return medication


def create_notes_df(dfs, filename=None):
    # filename indicating where embeddings are stored to load them
    # expects raw dfs and returns notes df with multiple texts per patient
    static_df = create_static_df(dfs)
    print("Reading notes from exams.csv")
    notes = dfs['exams'][['PatientID', 'TransplantationID', 'X', 'Befund', 'Art']].rename(
            columns={
                'PatientID': 'patient_id',
                'TransplantationID': 'transplant_id',
                'X': 'rel_days',
                'Befund': 'text',
                'Art': 'type',
            }
        )
    print(f"Found {len(notes)} texts")

    # Select and rename the relevant columns from clinical_assessment to match the structure of notes
    clinical_assessment_subset = dfs['clinical_assessment'][['PatientID', 'TransplantationID', 'OPDtime', 'BeurteilungAerztlich']].copy()
    clinical_assessment_subset.rename(
        columns={
            'PatientID': 'patient_id',
            'TransplantationID': 'transplant_id',
            'OPDtime': 'rel_days',
            'BeurteilungAerztlich': 'text'
        },
        inplace=True
    )

    clinical_assessment_subset['type'] = 'clinical_assessment'
    print(f"Loading {len(clinical_assessment_subset)} texts from clinical assessments")
    # Append the transformed clinical_assessment data to notes
    notes = pd.concat([notes, clinical_assessment_subset], ignore_index=True)
    notes = notes.dropna(subset=['text'])

    # remove patients not in static df
    notes = notes.merge(
        static_df[['patient_id', 'transplant_id']],
        how='inner',  # Inner join to keep only matching entries
        on=['patient_id', 'transplant_id']
    )

    print(f"Concatenated texts and deleted NaNs, final count: {len(notes)}")
    print(f"Average texts per patient: {len(notes) / len(static_df):.1f}")

    # Replace multiple <br>, ---- and ===== tags with a single whitespace in the notes
    notes['text'] = notes['text'].str.replace(r'(?i)(<br>\s*)+', ' ', regex=True)
    notes['text'] = notes['text'].str.replace(r'(=){2,}', '', regex=True)
    notes['text'] = notes['text'].str.replace(r'[-]{2,}', ' ', regex=True)
    notes['text'] = notes['text'].str.replace(r'\s+', ' ', regex=True)


    if filename is not None:
        loaded_embeddings = np.load(filename, allow_pickle=True)
        notes['embeddings'] = list(loaded_embeddings)

    notes['rel_days'] = notes['rel_days'].astype(str).str.replace(',', '.').astype(float).astype(int)

    return notes


def create_ts_data(vitals_ca, vitals_lab=None, medication=None, merge_lab=True, merge_med=True, static_df=None):
    ts_data = vitals_ca.copy()

    if merge_lab:
        assert static_df is not None

        lab_filtered = vitals_lab[vitals_lab['description'].isin(['KreatininHP', 'LeukoEB', 'CRPHP', 'ProteinCSU', 'AlbuminKSU'])].copy()
        lab_filtered['value'] = pd.to_numeric(lab_filtered['value'].astype(str).str.replace(',', '.'), errors='coerce')
        lab_filtered = lab_filtered.dropna(subset=['value'])
        lab_filtered['unit'] = lab_filtered['unit'].astype(str).str.lower()

        # Convert CRPHP mg/l to mg/dl
        crphp_mask = lab_filtered['description'] == 'CRPHP'
        mg_l_mask = crphp_mask & (lab_filtered['unit'] == 'mg/l')
        lab_filtered.loc[mg_l_mask, 'value'] = lab_filtered.loc[mg_l_mask, 'value'] / 10
        lab_filtered.loc[crphp_mask, 'unit'] = 'mg/dl'

        # Apply filters
        k_mask = (lab_filtered['description'] == 'KreatininHP') & (lab_filtered['unit'].isin(['mg/dl', 'mg/dl'])) & (lab_filtered['value'] <= 50)
        leu_mask = (lab_filtered['description'] == 'LeukoEB') & (lab_filtered['unit'] == '/nl') & (lab_filtered['value'] <= 5000)
        crphp_mask = (lab_filtered['description'] == 'CRPHP') & (lab_filtered['unit'] == 'mg/dl') & (lab_filtered['value'] <= 200)
        protein_mask = (lab_filtered['description'] == 'ProteinCSU') & (lab_filtered['unit'] == 'mg/l') & (lab_filtered['value'] <= 2000)
        alb_mask = (lab_filtered['description'] == 'AlbuminKSU') & (lab_filtered['value'] <= 5000)
        lab_filtered = lab_filtered[k_mask | leu_mask | crphp_mask | protein_mask | alb_mask]

        lab_pivot = lab_filtered.pivot_table(
            index=['patient_id', 'transplant_id', 'rel_days'],
            columns='description',
            values='value',
            aggfunc='first'
        ).reset_index()

        lab_pivot = lab_pivot.rename(columns={'KreatininHP': 'creatinine', 'LeukoEB': 'leukocyte', 'CRPHP': 'crphp', 'ProteinCSU': 'proteinuria', 'AlbuminKSU': 'acr'})
        ts_data = pd.merge(ts_data, lab_pivot, on=['patient_id', 'transplant_id', 'rel_days'], how='outer')

        # merge to get age and gender 
        ts_data = pd.merge(
            ts_data, 
            static_df[['patient_id', 'age', 'gender']], 
            on='patient_id', 
            how='left'
        )

        # Fallback, if age is missing, use 50; if gender is missing, assume 'm'
        ts_data['age'] = ts_data['age'].fillna(50)
        ts_data['gender'] = ts_data['gender'].fillna('m')

        # CALCULATE eGFR, follows formula from https://pmc.ncbi.nlm.nih.gov/articles/PMC3321332/
        # Convert mg/dL to µmol/L
        # Common conversion factor for creatinine: 1 mg/dL ~ 88.4 µmol/L
        ts_data['creatinine_umol'] = ts_data['creatinine'] * 88.4

        # Initialize egfr column as NaN
        ts_data['egfr'] = np.nan

        # Define masks
        female_mask = ts_data['gender'].eq('w')
        male_mask   = ts_data['gender'].eq('m')

        fe_lt62 = female_mask & (ts_data['creatinine_umol'] < 62)
        fe_ge62 = female_mask & (ts_data['creatinine_umol'] >= 62)

        ma_lt80 = male_mask & (ts_data['creatinine_umol'] < 80)
        ma_ge80 = male_mask & (ts_data['creatinine_umol'] >= 80)

        # Formula:
        # Female < 62: eGFR = 144 x (Cr/61.6)^-0.329 x (0.993)^Age
        ts_data.loc[fe_lt62, 'egfr'] = (144 * (ts_data.loc[fe_lt62, 'creatinine_umol'] / 61.6) ** -0.329 * (0.993 ** ts_data.loc[fe_lt62, 'age']))

        # Female >= 62: eGFR = 144 x (Cr/61.6)^-1.209 x (0.993)^Age
        ts_data.loc[fe_ge62, 'egfr'] = (144 * (ts_data.loc[fe_ge62, 'creatinine_umol'] / 61.6) ** -1.209 * (0.993 ** ts_data.loc[fe_ge62, 'age']))

        # Male < 80: eGFR = 141 x (Cr/79.2)^-0.411 x (0.993)^Age
        ts_data.loc[ma_lt80, 'egfr'] = (141 * (ts_data.loc[ma_lt80, 'creatinine_umol'] / 79.2) ** -0.411 * (0.993 ** ts_data.loc[ma_lt80, 'age']))

        # Male >= 80: eGFR = 141 x (Cr/79.2)^-1.209 x (0.993)^Age
        ts_data.loc[ma_ge80, 'egfr'] = (141 * (ts_data.loc[ma_ge80, 'creatinine_umol'] / 79.2) ** -1.209 * (0.993 ** ts_data.loc[ma_ge80, 'age']))

        # If creatinine is NaN, ensure eGFR is NaN
        ts_data.loc[ts_data['creatinine'].isna(), 'egfr'] = np.nan

        # Drop the helper column 
        ts_data.drop(columns='creatinine_umol', inplace=True)

    if merge_med:
        meds_of_interest = ['Tacrolimus', 'Methylprednisolon', 'Ciclosporin']
        meds_filtered = medication[medication['description'].isin(meds_of_interest)].copy()
        meds_filtered = meds_filtered[meds_filtered['unit'] == 'mg']

        meds_filtered['ddd'] = meds_filtered['ddd'].astype(str).str.replace(',', '.')
        meds_filtered['ddd'] = pd.to_numeric(meds_filtered['ddd'], errors='coerce')
        meds_filtered = meds_filtered.dropna(subset=['ddd'])

        # Remove entries where ddd > 2000, these are erroneous rows
        meds_filtered = meds_filtered[meds_filtered['ddd'] <= 2000]

        meds_filtered['rel_days'] = meds_filtered.apply(
            lambda r: range(r['start'], r['end'] + 1), axis=1
        )
        meds_expanded_df = meds_filtered.explode('rel_days').drop(columns=['start', 'end'])

        # Pivot to get medications in columns
        meds_pivot = meds_expanded_df.pivot_table(
            index=['patient_id', 'transplant_id', 'rel_days'],
            columns='description',
            values='ddd',
            aggfunc='sum'
        ).reset_index()

        ts_data = pd.merge(
            ts_data,
            meds_pivot,
            on=['patient_id', 'transplant_id', 'rel_days'],
            how='left'
        )
        ts_data[meds_of_interest] = ts_data[meds_of_interest].fillna(0)

    # Sorting
    ts_data = ts_data.sort_values(by=['patient_id', 'rel_days']).reset_index(drop=True)
    return ts_data


class NephroCAGEDataset(Dataset):
    def __init__(self, static_df: pd.DataFrame, ts_data: pd.DataFrame, notes_df: pd.DataFrame, biopsy_df: pd.DataFrame):

        self.static_df = static_df.copy()

        # create graft loss labels
        self.labels = self.static_df[['patient_id', 'loss_rel_days', 'death_rel_days']].copy()
        self.labels['graft_loss_label'] = self.labels['loss_rel_days'].notna().astype(int)
        self.labels['death_label'] = self.labels['death_rel_days'].notna().astype(int)

        # Handle missing values in static df
        # Use -1/Unknown to indicate missing value
        for col in CONFIG['static_numerical_cols']:
            self.static_df[col] = self.static_df[col].fillna(-1)
            #median = self.static_df[col].median()
            #self.static_df[col] = self.static_df[col].fillna(median)
        for col in CONFIG['static_categorical_cols']:
            self.static_df[col] = self.static_df[col].fillna('Unknown')

        # Encode categorical variables
        self.label_encoders = {}
        for col in CONFIG['static_categorical_cols']:
            le = LabelEncoder()
            self.static_df[col] = le.fit_transform(self.static_df[col])
            self.label_encoders[col] = le

        self.categorical_cardinalities = [
            len(le.classes_) for le in self.label_encoders.values()
        ]

        # Normalize numerical variables
        self.scaler = StandardScaler()
        self.static_df[CONFIG['static_numerical_cols']] = self.scaler.fit_transform(
            self.static_df[CONFIG['static_numerical_cols']]
        )

        self.ts_data = ts_data.copy()
        # value mask for time series data
        self.ts_data_value_mask = self.ts_data[['patient_id', 'rel_days']].copy()
        self.ts_data_value_mask[CONFIG['ts_features']] = (~self.ts_data[CONFIG['ts_features']].isna()).astype(int)

        # replace NaNs with padding value
        self.ts_data[CONFIG['ts_features']] = self.ts_data[CONFIG['ts_features']].fillna(CONFIG['PADDING_VAL'])

        # Normalize
        self.ts_scaler = StandardScaler()
        self.ts_data[CONFIG['ts_features']] = self.ts_scaler.fit_transform(self.ts_data[CONFIG['ts_features']])

        self.notes_df = notes_df.copy()  # Included as a separate dataframe

        # Compute rejection label by BANFF categories 2 and 4
        biopsy_df['Banff 17 categorie'] = pd.to_numeric(biopsy_df['Banff 17 categorie'], errors='coerce')
        rejection_rows = biopsy_df[biopsy_df['Banff 17 categorie'].isin([2, 4])].copy()
        grouped_rejections = (rejection_rows.groupby('PatientID')['BXtime'].apply(list).reset_index(name='rej_rel_days_list'))
        
        # Convert to dictionary: { patient_id: [day1, day2, ...] }
        self.rej_dict = dict(zip(grouped_rejections['PatientID'], grouped_rejections['rej_rel_days_list']))
        
        ts_counts = self.ts_data.groupby('patient_id').size().reset_index(name='counts')
        valid_patient_ids = ts_counts[ts_counts['counts'] >= 10]['patient_id'].unique()

        notes_counts = self.notes_df.groupby('patient_id').size().reset_index(name='counts')
        valid_patient_ids_with_notes = notes_counts[notes_counts['counts'] > 0]['patient_id'].unique()

        # Intersect with the already-valid patients above
        valid_patient_ids = set(valid_patient_ids).intersection(valid_patient_ids_with_notes)
        valid_patient_ids = np.array(list(valid_patient_ids))  # turn back to NumPy array

        # Filter self.static_df, self.labels, and self.ts_data to include only valid_patient_ids
        self.static_df = self.static_df[self.static_df['patient_id'].isin(valid_patient_ids)].copy()
        self.labels = self.labels[self.labels['patient_id'].isin(valid_patient_ids)].copy()
        self.ts_data = self.ts_data[self.ts_data['patient_id'].isin(valid_patient_ids)].copy()
        self.notes_df = self.notes_df[self.notes_df['patient_id'].isin(valid_patient_ids)].copy()

        # Get unique patient_ids
        self.patient_ids = self.static_df['patient_id'].unique().astype(int)
        if len(self.patient_ids) != len(self.static_df):
            raise ValueError("Duplicate patients in static df")

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]

        # Static Features
        static_row = self.static_df[self.static_df['patient_id'] == patient_id]
        if static_row.empty:
            raise ValueError(f"No static data found for patient_id: {patient_id}")
        static_features = static_row.drop(columns=['patient_id']).iloc[0]

        # Split categorical and numerical features
        categorical_features = torch.tensor(static_features[CONFIG['static_categorical_cols']].values.astype(np.int64))
        numerical_features = torch.tensor(static_features[CONFIG['static_numerical_cols']].values.astype(np.float32))

        # time-series feature tensor
        ts_data = self.ts_data[self.ts_data['patient_id'] == patient_id]
        ts_features = torch.tensor(ts_data[CONFIG['ts_features']].values.astype(np.float32))
        timesteps = torch.tensor(ts_data['rel_days'].values.astype(np.int64))

        # Value mask for time-series data
        value_mask = self.ts_data_value_mask[self.ts_data_value_mask['patient_id'] == patient_id]
        value_mask = torch.tensor(value_mask[CONFIG['ts_features']].values.astype(np.int64))

        seq_len = ts_features.size(0)

        # Handle notes embeddings and timesteps
        notes_rows = self.notes_df[self.notes_df['patient_id'] == patient_id]
        notes_timesteps = torch.tensor(notes_rows['rel_days'].values.astype(np.int64))
        if 'embeddings' in self.notes_df.columns and not notes_rows.empty:
            notes_embeddings = torch.tensor(np.stack(notes_rows['embeddings'].values), dtype=torch.float32)
        else:
            # Return an empty embedding of shape (0, embedding_dim)
            notes_embeddings = None  # Set to None if embeddings are not available

        # get labels
        labels = self.labels[self.labels['patient_id'] == patient_id]
        graft_loss_label = torch.tensor(labels['graft_loss_label'].values.astype(np.float32))
        death_label = torch.tensor(labels['death_label'].values.astype(np.float32))
        loss_rel_days = torch.tensor(labels['loss_rel_days'].values.astype(np.float32))
        death_rel_days = torch.tensor(labels['death_rel_days'].values.astype(np.float32))

        # rejection labels
        rej_rel_days_list = self.rej_dict.get(patient_id, [])
        if len(rej_rel_days_list) == 0:
            rej_rel_days_list = None

        sample = {
            'patient_id': patient_id,
            'static_categorical_features': categorical_features,
            'static_numerical_features': numerical_features,
            'ts_features': ts_features,
            'timesteps': timesteps,
            'value_mask': value_mask, # indicates missing values in the time series data
            'graft_loss_label': graft_loss_label,
            'loss_rel_days': loss_rel_days,
            'death_label': death_label,
            'death_rel_days': death_rel_days,
            'rej_rel_days': rej_rel_days_list,  # list of days of rejection
            'seq_len': seq_len,  # sequence length of time series data for this patient
            'notes_embeddings': notes_embeddings,  # variable length per patient around 10-100 embedded notes
            'notes_timesteps': notes_timesteps,  # corresponding to notes relative days since transplant
        }

        return sample


def collate_fn(batch):
    collated_batch = {}

    # Collect sequence lengths
    seq_lengths = torch.tensor([item['seq_len'] for item in batch], dtype=torch.long)
    # Check if notes embeddings are available in the batch
    has_notes = True
    if batch[0]['notes_embeddings'] is None:
        has_notes = False
    if has_notes:
        notes_lengths = torch.tensor([item['notes_embeddings'].shape[0] for item in batch], dtype=torch.long)
        max_notes_len = notes_lengths.max().item()
    else:
        notes_lengths = None
        max_notes_len = None

    # Find the maximum sequence length in the batch
    max_seq_len = seq_lengths.max().item()

    for key in batch[0].keys():
        data = [item[key] for item in batch]

        if key in ['ts_features', 'timesteps', 'value_mask']:
            # Pad 'ts_features', 'timsteps', and 'value_mask' to make all sequences the same length
            padding_value = CONFIG['PADDING_VAL'] if key != 'value_mask' else 0
            collated_batch[key] = pad_sequence(data, batch_first=True, padding_value=padding_value)
        elif key == 'notes_embeddings' or key == 'notes_timesteps':
            # Pad notes embeddings and timesteps
            if has_notes:
                collated_batch[key] = pad_sequence(data, batch_first=True, padding_value=0)
        elif key == 'seq_len':
            # Already collected sequence lengths
            collated_batch[key] = seq_lengths
        elif all(isinstance(d, torch.Tensor) and d.shape == data[0].shape for d in data):
            # Stack features that have consistent shapes
            collated_batch[key] = torch.stack(data)
        else:
            # Collect non-tensor or variable-length data as-is
            collated_batch[key] = data

    # Create the mask based on sequence lengths
    batch_size = len(seq_lengths)
    mask = torch.arange(max_seq_len).expand(batch_size, max_seq_len) < seq_lengths.unsqueeze(1)
    collated_batch['mask'] = mask  # Shape: (batch_size, max_seq_len)

    # Create the mask for notes embeddings
    if has_notes:
        notes_mask = torch.arange(max_notes_len).expand(batch_size, max_notes_len) < notes_lengths.unsqueeze(1)
        collated_batch['notes_mask'] = notes_mask  # Shape: (batch_size, max_notes_len)
    else:
        collated_batch['notes_mask'] = None

    return collated_batch