'''Adapted from https://github.com/sysu-yanglab/TDimpute
'''

import tensorflow as tf
tf.reset_default_graph()
import pandas as pd
import numpy as np
import pickle as pkl
import time
import os
from sklearn import preprocessing


def train(drop_prob, dataset_test, normal_scale=1.001, sav=True, checkpoint_file='default.ckpt'):

    input_image = tf.placeholder(tf.float32, batch_shape_input, name='input_image')
    is_training = tf.placeholder(tf.bool)
    scale = 0.
    with tf.variable_scope('FCN') as scope:
        fc_1 = tf.layers.dense(inputs=input_image, units=4000,
                               kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=scale))
        fc_1_out = tf.nn.sigmoid(fc_1)
        fc_1_dropout = tf.layers.dropout(inputs=fc_1_out, rate=drop_prob, training=is_training)
        fc_2_dropout = tf.layers.dense(inputs=fc_1_dropout, units=miRNA_size)  # 46744
        fc_2_out = tf.nn.sigmoid(fc_2_dropout)  # fc_2_dropout 
        reconstructed_image = fc_2_out  # fc_2_dropout

    original = tf.placeholder(tf.float32, batch_shape_output, name='original')
    loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(reconstructed_image, original))))
    l2_loss = tf.losses.get_regularization_loss()
    optimizer = tf.train.AdamOptimizer(lr).minimize(loss + l2_loss)

    init = tf.global_variables_initializer()
    saver = tf.train.import_meta_graph(checkpoint_file + '.meta') 
    start = time.time()
    loss_val_list_train = 0
    loss_val_list_test = 0
    loss_test = 0
    loss_test_pretrain = 0

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as session:
        session.run(init)
        print(("Loading variables from '%s'." % checkpoint_file))
        saver.restore(session, checkpoint_file)
        print('restored')

        ############### test the pretrain model for target dataset
        dataset_test = np.asarray(dataset_test).astype("float32")
        print('dataset_test', dataset_test.shape)
        reconstruct = session.run(reconstructed_image, feed_dict={input_image: dataset_test, is_training: False})

    end = time.time()
    el = end - start
    print(("Time elapsed %f" % el))
    return (loss_val_list_train, loss_val_list_test, loss_test, loss_test_pretrain, reconstruct)

def load_data(modalities_to_use):
    if 'RNA' in modalities_to_use:
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

    if 'miRNA' in modalities_to_use:
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

    if 'DNA' in modalities_to_use:
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

    # Now data preprocessing is done, can change from names to mod1 and mod2
    # We use mod1 to impute mod2
    if modalities_to_use[0] == 'RNA':
        mod1_df = rna_df.copy()
    elif modalities_to_use[0] == 'DNA':
        mod1_df = dna.copy()
    elif modalities_to_use[0] == 'miRNA':
        mod1_df = mirna_df.copy()
    else:
        print('Modality to impute from not recognised')
    
    if modalities_to_use[1] == 'RNA':
        mod2_df = rna_df.copy()
    elif modalities_to_use[1] == 'DNA':
        mod2_df = dna.copy()
    elif modalities_to_use[1] == 'miRNA':
        mod2_df = mirna_df.copy()
    else:
        print('Modality to impute not recognised')
    return mod1_df, mod2_df

#############################################################################################################
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

modalities_to_use = ['RNA', 'miRNA']

mod1_df, mod2_df = load_data(modalities_to_use)

# Filter to just patients we want to impute - CHANGE THIS DEPENDING ON TASK
with open('modality_patient_lists/just_rnadna_patients.txt', 'rb') as handle:
    patients_to_predict = pkl.load(handle)
mod1_df = mod1_df[mod1_df.index.isin(patients_to_predict)]

RNA_size = mod1_df.shape[1]
miRNA_size = mod2_df.shape[1]
print(RNA_size, miRNA_size)

save_ckpt = True
test_data = mod1_df.values
print('Data used for prediction: ', test_data.shape)

lr = 0.0001
drop_prob = 0.
batch_shape_input = (None, RNA_size)
batch_shape_output = (None, miRNA_size)
print('input shape: ', batch_shape_input, ', output shape: ', batch_shape_output)
loss_val_list_train, loss_val_list_test, loss_test, loss_test_pretrain, reconstruct = train(drop_prob,
                                                                                            test_data,
                                                                                            sav=save_ckpt,
                                                                                            checkpoint_file="TDimpute/TDimpute_checkpoint/general_model_for_rnamirna_pancan.ckpt") 

# only use column names from miRNA file
miRNA_txt = pd.DataFrame(reconstruct, index=mod1_df.index, columns=mod2_df.columns)
miRNA_txt.to_csv('data/tdimpute_omiembed_mirna_imputed_09.11.csv')
