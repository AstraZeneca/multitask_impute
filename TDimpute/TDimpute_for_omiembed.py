'''Adapted from https://github.com/sysu-yanglab/TDimpute
'''

import tensorflow as tf   #using old code from v1 of tensorflow
import pandas as pd
import numpy as np
import time
import os
from sklearn import preprocessing

def get_next_batch(dataset1, batch_size_1, step, ind):
    start = step * batch_size_1
    end = ((step + 1) * batch_size_1)
    sel_ind = ind[start:end]
    newdataset1 = dataset1[sel_ind, :]
    return newdataset1

def train(drop_prob, source_data, dataset_train, dataset_test, normal_scale, sav=True, checkpoint_file='default.ckpt'):
    target_data = dataset_train
    dataset_train = source_data
    input_image = tf.placeholder(tf.float32, batch_shape_input, name='input_image')
    is_training = tf.placeholder(tf.bool)
    scale = 0.
    with tf.variable_scope('autoencoder') as scope:
        fc_1 = tf.layers.dense(inputs=input_image, units=4000,
                               kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=scale))
        fc_1_out = tf.nn.sigmoid(fc_1)
        fc_1_dropout = tf.layers.dropout(inputs=fc_1_out, rate=drop_prob, training=is_training)
        fc_2_dropout = tf.layers.dense(inputs=fc_1_dropout, units=mod2_size)  # 46744
        fc_2_out = tf.nn.sigmoid(fc_2_dropout)
        reconstructed_image = fc_2_out  # fc_2_dropout

    original = tf.placeholder(tf.float32, batch_shape_output, name='original')
    print('original shape', original.shape, 'reconstructed shape', reconstructed_image.shape)
    loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(reconstructed_image, original))))
    l2_loss = tf.losses.get_regularization_loss()
    optimizer = tf.train.AdamOptimizer(lr).minimize(loss + l2_loss)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    start = time.time()
    loss_val_list_train = 0
    loss_val_list_test = 0

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as session:
        session.run(init)

        if sav:
            ############ transfer learning -> train on the source dataset
            dataset_size_train = dataset_train.shape[0]
            batch_size = 128
            num_epochs = 300
            num_iters = (num_epochs * dataset_size_train) // batch_size
            print("Num iters:", num_iters)
            ind_train = []
            for i in range(num_epochs):
                ind_train = np.append(ind_train, np.random.permutation(np.arange(dataset_size_train)))
            ind_train = np.asarray(ind_train).astype("int32")

            total_cost_train = 0.
            num_batchs = dataset_size_train // batch_size  # "//" => int()
            for step in range(num_iters):
                temp = get_next_batch(dataset_train, batch_size, step, ind_train)
                train_batch = np.asarray(temp).astype("float32")
                train_loss_val, _ = session.run([loss, optimizer],
                                                feed_dict={input_image: train_batch[:, mod2_size:],
                                                           original: train_batch[:, :mod2_size],
                                                           is_training: True})
                loss_val_list_train = np.append(loss_val_list_train, train_loss_val)
                total_cost_train += train_loss_val

                print_epochs = 10
                if step % (num_batchs * print_epochs) == 0:  # by epoch, num_batchs * batch_size = dataset_train_size
                    ############### test section  ###########################
                    dataset_test = np.asarray(dataset_test).astype("float32")

                    test_loss_val = session.run(loss, feed_dict={input_image: dataset_test[:, mod2_size:],
                                                                 original: dataset_test[:, :mod2_size],
                                                                 is_training: False})

                    reconstruct = session.run(reconstructed_image,
                                              feed_dict={input_image: dataset_test[:, mod2_size:], is_training: False})
                    nz = dataset_test[:, :mod2_size].shape[0] * dataset_test[:, :mod2_size].shape[1]
                    diff_mat = ((reconstruct - dataset_test[:, :mod2_size]) * normal_scale) ** 2
                    loss_test = np.sqrt(np.sum(diff_mat) / nz)

                    print('RMSE loss at pretrain: ', step, "/", num_iters,
                          total_cost_train / (num_batchs * print_epochs),
                          test_loss_val, loss_test)
                    total_cost_train = 0.

            save_path = saver.save(session, checkpoint_file)  # " checkpoints/i{}_l{}.ckpt".format(counter, lstm_size))
            print(("Model saved in file: %s" % save_path))

        else:
            print(("Loading variables from '%s'." % checkpoint_file))
            saver.restore(session, checkpoint_file)
            print('restored')

        ############### test the pretrain model for target dataset
        dataset_test = np.asarray(dataset_test).astype("float32")
        reconstruct = session.run(reconstructed_image,
                                  feed_dict={input_image: dataset_test[:, mod2_size:], is_training: False})
        nz = dataset_test[:, :mod2_size].shape[0] * dataset_test[:, :mod2_size].shape[1]
        diff_mat = ((reconstruct - dataset_test[:, :mod2_size]) * normal_scale) ** 2
        loss_test_pretrain = np.sqrt(np.sum(diff_mat) / nz)
        print('RMSE loss at pretrain: ', loss_test_pretrain)

        ################## transfer learning -> in the target dataset
        dataset_train = target_data
        batch_size = 16
        num_epochs = 150

        dataset_size_train = dataset_train.shape[0]
        dataset_size_test = dataset_test.shape[0]
        print("Dataset size for training:", dataset_size_train)
        print("Dataset size for test:", dataset_size_test)
        num_iters = (num_epochs * dataset_size_train) // batch_size
        print("Num iters:", num_iters)
        ind_train = []
        for i in range(num_epochs):
            ind_train = np.append(ind_train, np.random.permutation(np.arange(dataset_size_train)))
        ind_train = np.asarray(ind_train).astype("int32")

        total_cost_train = 0.
        num_batchs = dataset_size_train // batch_size  # "//" => int()
        for step in range(num_iters):
            temp = get_next_batch(dataset_train, batch_size, step, ind_train)
            train_batch = np.asarray(temp).astype("float32")
            train_loss_val, _ = session.run([loss, optimizer],
                                            feed_dict={input_image: train_batch[:, mod2_size:],
                                                       original: train_batch[:, :mod2_size],
                                                       is_training: True})

            loss_val_list_train = np.append(loss_val_list_train, train_loss_val)
            total_cost_train += train_loss_val

            print_epochs = 10
            if step % (num_batchs * print_epochs) == 0:  # by epoch, num_batchs * batch_size = dataset_train_size
                ############### test section  ###########################
                dataset_test = np.asarray(dataset_test).astype("float32")
                test_loss_val = session.run(loss, feed_dict={input_image: dataset_test[:, mod2_size:],
                                                             original: dataset_test[:, :mod2_size],
                                                             is_training: False})
                loss_val_list_test = np.append(loss_val_list_test, test_loss_val)

                reconstruct = session.run(reconstructed_image,
                                          feed_dict={input_image: dataset_test[:, mod2_size:], is_training: False})
                nz = dataset_test[:, :mod2_size].shape[0] * dataset_test[:, :mod2_size].shape[1]
                diff_mat = ((reconstruct - dataset_test[:, :mod2_size]) * normal_scale) ** 2
                loss_test = np.sqrt(np.sum(diff_mat) / nz)

                print('RMSE loss by train_data_size: ', step, "/", num_iters,
                      total_cost_train / (num_batchs * print_epochs),
                      test_loss_val, loss_test)
                # print('RMSE loss by train_data_size: ', step, "/", num_iters, total_cost_train / num_batchs,
                #       total_cost_validation / num_batchs)
                #                 print('new loss: ', step, "/", num_iters, train_loss_val, valid_loss_val)
                total_cost_train = 0.
                total_cost_validation = 0.

        ####final test
        ############### test section  ###########################
        dataset_test = np.asarray(dataset_test).astype("float32")

        test_loss_val = session.run(loss, feed_dict={input_image: dataset_test[:, mod2_size:],
                                                     original: dataset_test[:, :mod2_size],
                                                     is_training: False})
        loss_val_list_test = np.append(loss_val_list_test, test_loss_val)

        reconstruct = session.run(reconstructed_image,
                                  feed_dict={input_image: dataset_test[:, mod2_size:], is_training: False})
        nz = dataset_test[:, :mod2_size].shape[0] * dataset_test[:, :mod2_size].shape[1]
        diff_mat = ((reconstruct - dataset_test[:, :mod2_size]) * normal_scale) ** 2
        loss_test = np.sqrt(np.sum(diff_mat) / nz)

        print('RMSE loss by train_data_size: ', step, "/", num_iters, total_cost_train / (num_batchs * print_epochs),
              test_loss_val, loss_test)
        # print('RMSE loss by train_data_size: ', step, "/", num_iters, total_cost_train / num_batchs,
        #       total_cost_validation / num_batchs)
        #                 print('new loss: ', step, "/", num_iters, train_loss_val, valid_loss_val)

    end = time.time()
    el = end - start
    print(("Time elapsed %f" % el))
    return (loss_val_list_train, loss_val_list_test, loss_test, loss_test_pretrain, reconstruct)

def load_data(modalities_to_use):
    if 'RNA' in modalities_to_use:
        rna_df = pd.read_csv('HiSeqV2', sep='\t')
        rna_df.set_index('Sample', inplace=True)
        rna_df = rna_df.T
        with open('rna_5k_variable_genes.txt', 'rb') as handle:
            rna_top_genes = pkl.load(handle)    
        rna_df = rna_df[rna_top_genes]
        rna_df = rna_df.fillna(0).sort_index(axis=0)
        rna_subset_scaled = preprocessing.MinMaxScaler().fit_transform(rna_df) # Scale RNAseq data using zero-one normalization
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
        mirna_subset_scaled = preprocessing.MinMaxScaler().fit_transform(mirna_df) # Scale RNAseq data using zero-one normalization
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

if __name__ == "__main__":
    import getopt
    import sys
    import pickle as pkl
    import numpy as np
    print('Sys argv is:', sys.argv)

    ''' 
    #To get most variable genes and save gene list
    mod1_df = pd.read_csv('data/gbm_download/HiSeqV2', sep='\t')
    mod1_df.set_index('Sample', inplace=True)
    mod1_df = mod1_df.T
    num_mad_genes = 5000
    mad_genes = mod1_df.mad(axis=0).sort_values(ascending=False)
    top_mad_genes = mad_genes.iloc[0:num_mad_genes, ].index
    print(top_mad_genes.shape)
    mod1_df = mod1_df.loc[:, top_mad_genes]
    gene_names = mod1_df.columns.tolist()
    with open("rna_5k_variable_genes.txt", "wb") as fp:   
        pkl.dump(gene_names, fp)    
    
    # To preprocess DNA and get most variable probes
    dna = pd.read_csv('GDC-PANCAN.methylation450.tsv.gz', sep='\t', index_col=0)
    dna = dna.T    
    #Drop probes with more than 90% missing
    dna = dna.dropna(thresh=dna.shape[0]*0.1,how='all',axis=1)
    print('Without probes with >90% missing: ', dna.shape)

    #Drop probes on Y chromosome
    y_chromosome_probes = pd.read_csv('OmiEmbed/anno/B_anno.csv')
    y_chromosome_probes = y_chromosome_probes[y_chromosome_probes['CHR'] == 'Y']['IlmnID'].tolist()
    y_chromosome_probes = list(set(y_chromosome_probes) & set(dna.columns))
    dna = dna.drop(y_chromosome_probes, axis=1)
    print('Remove probes on Y chromosomes: ', dna.shape)
    num_mad_genes = 5000
    mad_genes = dna.mad(axis=0).sort_values(ascending=False)
    top_mad_genes = mad_genes.iloc[0:num_mad_genes, ].index
    print(top_mad_genes.shape)
    dna = dna.loc[:, top_mad_genes]
    gene_names = dna.columns.tolist()
    with open("dna_5k_variable_genes.txt", "wb") as fp:   
        pkl.dump(gene_names, fp)
    
    '''

    # Load lists of patients with each combination of modalities 
    with open('modality_patient_lists/all_modalities_patients.txt', 'rb') as handle:
        all_modalities_patients = pkl.load(handle)
    with open('modality_patient_lists/just_dnamirna_patients.txt', 'rb') as handle:
        just_dnamirna_patients = pkl.load(handle)
    with open('modality_patient_lists/just_rnadna_patients.txt', 'rb') as handle:
        just_rnadna_patients = pkl.load(handle)
    with open('modality_patient_lists/just_rnamirna_patients.txt', 'rb') as handle:
        just_rnamirna_patients = pkl.load(handle)
    with open('modality_patient_lists/dnamirna_intersection.txt', 'rb') as handle:
        dnamirna_intersection = pkl.load(handle)
    with open('modality_patient_lists/rnadna_intersection.txt', 'rb') as handle:
        rnadna_intersection = pkl.load(handle)
    with open('modality_patient_lists/rnamirna_intersection.txt', 'rb') as handle:
        rnamirna_intersection = pkl.load(handle)

    modalities_to_use = ['RNA', 'DNA'] # we are imputing the 2nd modality using the 1st

    # Load data for required modalities
    mod1_df, mod2_df = load_data(modalities_to_use)
    print('Loaded data. Modality 1 shape is ', mod1_df.shape, ' , modality 2 shape is ', mod2_df.shape)
    print(mod1_df.head())
    print(mod2_df.head())

    # Filter to correct patients using lists of different modalities
    complete_data_patients = rnadna_intersection.copy() # change this depending on task

    complete_data_mod1 = mod1_df[mod1_df.index.isin(complete_data_patients)]
    complete_data_mod2 = mod2_df[mod2_df.index.isin(complete_data_patients)]
    mod1_for_impute = mod1_df[~mod1_df.index.isin(complete_data_patients)]
    print('Complete training data. Modality 1 shape is ', complete_data_mod1.shape, ' , modality 2 shape is ', complete_data_mod2.shape)
    print('Data to impute once model is trained. Modality 1 shape is ', mod1_for_impute.shape)
    
    # merge two modalities together
    merged_data = pd.merge(complete_data_mod1, complete_data_mod2, left_index=True, right_index=True)
    print('Shape of merged data: ', merged_data.shape)

    df = merged_data.values
    col = merged_data.columns
    columns_ind = merged_data.columns.tolist()
    rows_ind = merged_data.index

    RNA_DNA_txt = merged_data.copy()
    cancer_names = ['pancan'] 
    mod1_size = complete_data_mod1.shape[1] #no. columns
    mod2_size = complete_data_mod2.shape[1] #no. columns
    print('modality 1 size ', mod1_size, 'modality 2 size ', mod2_size)
    print('Ready for training')
    
    sample_size = 1 
    loss_list = np.zeros([1, 5, sample_size]); 
    loss_summary = np.zeros([1, 5])
    loss_list_pretrain = np.zeros([1, 5, sample_size]);    #for pretrain leave at 16
    loss_summary_pretrain = np.zeros([1, 5])
    cancer_c = 0

    for cancertype in cancer_names:
        perc = 0
        save_ckpt = True 
        for missing_perc in [0.3]: 	
            for sample_count in range(1, sample_size + 1):
                shuffle_cancer = merged_data.copy()
                print('name:', cancertype, ' missing rate:', missing_perc, ' data size:',shuffle_cancer.shape)
                ########################Create set for training and testing
                normal_scale = 1.001 
                aa = np.concatenate((shuffle_cancer.values[:, :mod1_size] / normal_scale, shuffle_cancer.values[:, mod1_size:]), axis=1)
                shuffle_cancer = pd.DataFrame(aa, index=shuffle_cancer.index, columns=shuffle_cancer.columns)
                RDNA = shuffle_cancer.values
                test_data = RDNA[0:int(RDNA.shape[0] * missing_perc), :] 
                train_data = RDNA[int(RDNA.shape[0] * missing_perc):, :]
                print('train datasize:', train_data.shape[0], ' test datasize: ', test_data.shape[0])

                ## 32 cancer datasets are combined as source domain and the remaining one cancer is considered as target domain.
                source_data = RNA_DNA_txt[~RNA_DNA_txt.index.isin(shuffle_cancer.index)]  # bool index, e.g. df[df.A>0]
                aa = np.concatenate((source_data.values[:, :mod1_size] / normal_scale, source_data.values[:, mod1_size:]), axis=1)
                source_data = pd.DataFrame(aa, index=source_data.index, columns=source_data.columns)

                lr = 0.001  
                feature_size = RDNA.shape[1]  
                drop_prob = 0.  #dropout probability
                batch_shape_input = (None, mod1_size)  
                batch_shape_output = (None, mod2_size)
                tf.reset_default_graph()
                print('Training')
                loss_val_list_train, loss_val_list_test, loss_test,loss_test_pretrain, reconstruct = train(drop_prob, source_data.values, train_data, test_data,
                                                                                        normal_scale, sav=save_ckpt, checkpoint_file="data/checkpoints/general_model_for_rnadna_"+ cancertype+'.ckpt')

                save_ckpt = False
                imputed_data = np.concatenate([reconstruct*normal_scale, train_data[:, :mod2_size]*normal_scale], axis=0) 
                print('Imputed...')
                RNA_txt = pd.DataFrame(imputed_data[:, :mod2_size], index=shuffle_cancer.index,
                                    columns=shuffle_cancer.columns[:mod2_size]) 
                RNA_txt.to_csv('data/tdimpute_omiembed_imputed_dnafromrna_02.11.csv')

                loss_list[cancer_c, perc, sample_count - 1] = loss_test  
                loss_list_pretrain[cancer_c, perc, sample_count - 1] = loss_test_pretrain  
            perc = perc + 1
        np.set_printoptions(precision=3)
        print(np.array([np.mean(loss_list[cancer_c, i, :]) for i in range(0, 5)]))
        print(np.array([np.mean(loss_list_pretrain[cancer_c, i, :]) for i in range(0, 5)]))
        loss_summary[cancer_c, :] = np.array([np.mean(loss_list[cancer_c, i, :]) for i in range(0, 5)])
        loss_summary_pretrain[cancer_c, :] = np.array([np.mean(loss_list_pretrain[cancer_c, i, :]) for i in range(0, 5)])
        cancer_c = cancer_c + 1

    print('RMSE by cancer (averaged by sampling times):')
    print(loss_summary)
    print(loss_summary_pretrain)
    




