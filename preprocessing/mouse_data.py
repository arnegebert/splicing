import pickle

with open('../data/five_tissue_data.pkl', 'rb') as f:
    data = pickle.load(f, encoding='latin1')
    print('hello')

# current: pprocess gets killed, probably due to too much mem usage

# dict with 8 keys, n = 94449
# identifiers = array from 0,..., n-1
# abs_psi_labels = array (n, 6) - see other file
# cv_train_test_ids = which samples where used in which cv split
# psis = (n, 2) .. more psi values??
# data_array = (n, 1357); min: -22.93, max: 301... perhaps feature values
# udc_labels = (n, 3) values from (0, 1)
# tissue type = (n, 10) one-hot encoding of tissue type per sample i guess
# cv_train_val_test_ids = dict 15 with further subdivision idc