import numpy as np
import pandas as pd
import pickle as pkl
from sklearn import preprocessing

def rename_pancancer(cohort, type_id):
    renaming_dict = {'TCGA Breast Cancer': 'Breast',
                'TCGA Lung Adenocarcinoma': 'LUAD',
                'TCGA Glioblastoma': 'Glioblastoma',
                'TCGA Kidney Clear Cell Carcinoma': 'Kidney Clear Cell',
                'TCGA Ovarian Cancer': 'Ovarian',
                'TCGA Endometrioid Cancer': 'Endometrioid',
                'TCGA Head and Neck Cancer': 'Head and Neck',
                'TCGA Thyroid Cancer': 'Thyroid',
                'TCGA Lung Squamous Cell Carcinoma': 'LUSC',
                'TCGA Prostate Cancer': 'Prostate',
                'TCGA Lower Grade Glioma': 'Lower Grade Glioma',
                'TCGA Stomach Cancer': 'Stomach',
                'TCGA Colon Cancer': 'Colon',
                'TCGA Liver Cancer': 'Liver',
                'TCGA Melanoma': 'Melanoma',
                'TCGA Bladder Cancer': 'Bladder',
                'TCGA Kidney Papillary Cell Carcinoma': 'Kidney Papillary Cell',
                'TCGA Cervical Cancer': 'Cervical',
                'TCGA Sarcoma': 'Sarcoma',
                'TCGA Acute Myeloid Leukemia': 'AML',
                'TCGA Esophageal Cancer': 'Esophageal',
                'TCGA Pancreatic Cancer': 'Pancreatic',
                'TCGA Pheochromocytoma & Paraganglioma': 'Pheochromocytoma & Paraganglioma',
                'TCGA Rectal Cancer': 'Rectal',
                'TCGA Testicular Cancer': 'Testicular',
                'TCGA thymoma': 'Thymoma',
                'TCGA Adrenocortical Cancer': 'Adrenocortical',
                'TCGA Kidney Chromophobe': 'Kidney Chromophobe',
                'TCGA Ocular melanomas': 'Ocular melanomas',
                'TCGA Uterine Carcinosarcoma': 'Uterine Carcinosarcoma',
                'TCGA Large B-cell Lymphoma': 'Large B-cell Lymphoma',
                'TCGA Bile Duct Cancer': 'Bile Duct',
                'TCGA Mesothelioma': 'Mesothelioma'
                }
    cancer_type = renaming_dict.get(str(cohort))
    if type_id == 11.0:
        cancer_type = None      
    return cancer_type

def load_original_data(): # Load original data for all 3 modalities and clinical data

    clinicals = pd.read_csv('data/gbm_download/PANCAN_clinicalMatrix', sep='\t')
    clinicals.set_index('sampleID', inplace=True)
    clinicals = clinicals[clinicals['sample_type_id'] != 11.0]
    clinicals['cancer_type'] = clinicals.apply(lambda x: rename_pancancer(x['_cohort'], x['sample_type_id']), axis=1)
    clinicals = clinicals.dropna(subset=['cancer_type'])
    print('Initial shape of clinicals data: ', clinicals.shape)

    rna_df = pd.read_csv('data/gbm_download/HiSeqV2', sep='\t')
    rna_df.set_index('Sample', inplace=True)
    rna_df = rna_df.T
    with open('rna_5k_variable_genes.txt', 'rb') as handle:
        rna_top_genes = pkl.load(handle)    
    rna_df = rna_df[rna_top_genes]
    rna_df = rna_df.fillna(0).sort_index(axis=0)
    rna_subset_scaled = preprocessing.MinMaxScaler().fit_transform(rna_df) 
    rna_subset_scaled = pd.DataFrame(rna_subset_scaled, columns=rna_df.columns, index=rna_df.index)
    print('Initial shape of RNA data: ', rna_subset_scaled.shape)
    rna_df = rna_subset_scaled.copy()

    mirna_df = pd.read_csv('GDC-PANCAN.mirna.tsv.gz', sep='\t', index_col=0)
    mirna_df = mirna_df.T
    # Change index names (remove letter on the end of patient ID)
    mirna_index_list = mirna_df.index.tolist()
    mirna_index_list = [x[:-1] for x in mirna_index_list]
    mirna_df['index_short'] = mirna_index_list
    mirna_df.drop_duplicates(subset='index_short', inplace=True)
    mirna_df.set_index('index_short', drop=True, inplace=True)
    mirna_df = mirna_df[~mirna_df.index.duplicated(keep='first')]
    mirna_df = np.log2(mirna_df + 1)
    mirna_df = mirna_df.fillna(0).sort_index(axis=0)
    mirna_subset_scaled = preprocessing.MinMaxScaler().fit_transform(mirna_df) 
    mirna_subset_scaled = pd.DataFrame(mirna_subset_scaled, columns=mirna_df.columns, index=mirna_df.index)
    print('Initial shape of miRNA data: ', mirna_subset_scaled.shape)
    mirna_df = mirna_subset_scaled.copy()

    dna = pd.read_csv('GDC-PANCAN.methylation450.tsv.gz', sep='\t', index_col=0)
    dna = dna.T      
    # filter to just top 5k probes
    with open('dna_5k_variable_genes.txt', 'rb') as handle:
        dna_top_genes = pkl.load(handle)    
    dna = dna[dna_top_genes]        
    # change index names
    dna_index_list = dna.index.tolist()
    dna_index_list = [x[:-1] for x in dna_index_list]
    dna['index_short'] = dna_index_list
    dna.drop_duplicates(subset='index_short', inplace=True)
    dna.set_index('index_short', drop=True, inplace=True)
    dna = dna[~dna.index.duplicated(keep='first')]  
    dna = dna.fillna(0).sort_index(axis=0)
    print('Initial shape of DNA data: ', dna.shape)

    return clinicals, rna_df, dna, mirna_df

def load_imputed_data(): # Load imputed data for all 3 modalities
    rna_path = 'data/tdimpute_omiembed_rna_imputed_09.11.csv'
    dna_path = 'data/tdimpute_omiembed_dna_imputed_08.11.csv'
    mirna_path = 'data/tdimpute_omiembed_mirna_imputed_09.11.csv' 

    imputed_rna = pd.read_csv(rna_path, index_col=0)
    imputed_dna = pd.read_csv(dna_path, index_col=0)
    imputed_mirna = pd.read_csv(mirna_path, index_col=0)

    print('Imputed RNA: ', imputed_rna.shape)
    print('Imputed DNA: ', imputed_dna.shape)
    print('Imputed miRNA: ', imputed_mirna.shape)

    return imputed_rna, imputed_dna, imputed_mirna

def get_common_patients(clinicals, rna, dna, mirna):

    # Get unique patients for each 
    clinicals = clinicals[~clinicals.index.duplicated(keep='first')]
    rna = rna[~rna.index.duplicated(keep='first')]
    dna = dna[~dna.index.duplicated(keep='first')]
    mirna = mirna[~mirna.index.duplicated(keep='first')]
    print('clinicals shape no duplicates', clinicals.shape)
    print('RNA shape no duplicates', rna.shape)
    print('DNA shape no duplicates', dna.shape)
    print('miRNA shape no duplicates', mirna.shape)
    # How many of each modality are in the clinical list
    rna = rna[rna.index.isin(clinicals.index.tolist())]
    dna = dna[dna.index.isin(clinicals.index.tolist())]
    mirna = mirna[mirna.index.isin(clinicals.index.tolist())]
    print('RNA shape in clinicals', rna.shape)
    print('DNA shape in clinicals', dna.shape)
    print('miRNA shape in clinicals', mirna.shape)    
    # Find patients with all 3 modalities and clinicals
    rna_dna_common_unique = pd.merge(rna, dna, left_index=True, right_index=True)
    print('rna dna common unique ', rna_dna_common_unique.shape)
    all_three_common_unique = pd.merge(rna_dna_common_unique, mirna, left_index=True, right_index=True)
    print('all three modalities (before clinicals) ', all_three_common_unique.shape)
    # Filter to these patients
    clinicals = clinicals[clinicals.index.isin(all_three_common_unique.index.tolist())]
    print('clinicals with all 3 modalities', clinicals.shape)
    # Get patients with clinical data for all three specific tasks (classification, regression, survival)
    clinicals = clinicals[['_OS', '_EVENT', 'cancer_type', 'age_at_initial_pathologic_diagnosis']] 
    clinicals['cancer_type'] = clinicals['cancer_type'].astype('category')
    clinicals['sample_type.samples'] = clinicals['cancer_type'].cat.codes
    clinicals.drop(['cancer_type'], axis=1, inplace=True)
    mean_age=clinicals['age_at_initial_pathologic_diagnosis'].mean()
    clinicals['age_at_initial_pathologic_diagnosis'].fillna(value=mean_age, inplace=True)
    print('Common patients for all 3 tasks (clinicals) before dropna: ', clinicals.shape)
    clinicals.dropna(axis=0, inplace=True)
    print('Common patients for all 3 tasks (clinicals): ', clinicals.shape)
    rna = rna[rna.index.isin(clinicals.index.tolist())]
    dna = dna[dna.index.isin(clinicals.index.tolist())]
    mirna = mirna[mirna.index.isin(clinicals.index.tolist())]   
    print('Final shapes (should be same): ', clinicals.shape, rna.shape, dna.shape, mirna.shape)

    return clinicals, rna, dna, mirna

def process_for_omiembed(clinicals, rna, dna, mirna):
    # Impute
    rna = rna.fillna(rna.mean())    
    dna = dna.fillna(dna.mean())
    mirna = mirna.fillna(mirna.mean()) 
    # Sort
    clinicals = clinicals.sort_index()
    rna = rna.sort_index()
    dna = dna.sort_index()
    mirna = mirna.sort_index()
    # Transpose
    rna = rna.T
    dna = dna.T
    mirna = mirna.T
    print('Modality shapes for omiembed: ', rna.shape, dna.shape, mirna.shape)
    # Get labels for each task
    clinicals_type = clinicals.drop(['_OS', '_EVENT', 'age_at_initial_pathologic_diagnosis'], axis=1)
    print('Final classification shape: ', clinicals_type.shape)
    clinicals_age = clinicals.drop(['_OS', '_EVENT', 'sample_type.samples'], axis=1) # to change regression task from age to STX1A expression
    print('Final regression shape: ', clinicals_age.shape)
    clinicals_survival = clinicals.drop(['sample_type.samples', 'age_at_initial_pathologic_diagnosis'], axis=1)
    clinicals_survival.columns = ['survival_T', 'survival_E']
    print('Final survival shape: ', clinicals_survival.shape)

    return clinicals_type, clinicals_age, clinicals_survival, rna, dna, mirna

if __name__ == "__main__":

    clinicals, rna, dna, mirna = load_original_data() # load original data
    imputed_rna, imputed_dna, imputed_mirna = load_imputed_data() # load imputed data

    # for each modality, add imputed patients onto end of original datset
    rna = pd.concat([rna, imputed_rna])
    dna = pd.concat([dna, imputed_dna])
    mirna = pd.concat([mirna, imputed_mirna])

    print('Full shape of RNA: ', rna.shape)
    print('Full shape of DNA: ', dna.shape)
    print('Full shape of miRNA: ', mirna.shape)

    clinicals, rna, dna, mirna = get_common_patients(clinicals, rna, dna, mirna)
    clinicals_type, clinicals_age, clinicals_survival, rna, dna, mirna = process_for_omiembed(clinicals, rna, dna, mirna)

    # Save
    clinicals_type.to_csv('data/imputed_dataset/labels.tsv', sep='\t')
    clinicals_age.to_csv('data/imputed_dataset/values.tsv', sep='\t')
    clinicals_survival.to_csv('data/imputed_dataset/survival.tsv', sep='\t')
    rna.to_csv('data/imputed_dataset/rna.tsv', sep='\t')
    dna.to_csv('data/imputed_dataset/dna.tsv', sep='\t')
    mirna.to_csv('data/imputed_dataset/mirna.tsv', sep='\t')
