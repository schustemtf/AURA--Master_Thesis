CONFIG = {
    'static_categorical_cols': ['gender', 'underlying_disease', 'blood_group', 'gender_donor', 'donor_bloodgroup', 'type_of_donation'],
    
    'static_numerical_cols': ['age', 'number_dialyses', 'cold_ischemia_time', 'age_donor', 'pirche_score', 'mma_broad', 'mmb_broad', 'mmdr_broad','mm_broad'],
    
    'ts_features' : ['bp_sys', 'bp_dia', 'weight', 'urine_volume', 'hr', 'temperature', 'diuresis_time', 
                     'creatinine', 'leukocyte', 'proteinuria', 'crphp', 'egfr', 'acr', 'Tacrolimus', 'Methylprednisolon', 'Ciclosporin'],

    'static_embedding_dim': 16, # embedding size of each feature, categorical or numerical
    'static_output_dim': 256, # final static embedding dimension after last ff layer
    'lstm_hidden_size': 512, # hidden size in lstms
    'lstm_num_layers': 2,
    'num_heads': 2,
    'PADDING_VAL': 0,
    'notes_embedding_dim': 1024, # embedding dim of clinical notes
}