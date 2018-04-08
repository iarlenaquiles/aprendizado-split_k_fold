import numpy as np

def slit_k_fold(n_elem, n_splits, shuffle, seed):
    total = [ i for i in range(n_elem)]
    
    if shuffle:
        np.random.shuffle(total)
    if seed:
        np.random.seed(seed)
    
    fold_sizes = (n_elem // n_splits) * np.ones(n_splits, dtype=np.int)
    fold_sizes[:n_elem % n_splits] += 1
    
    idx_train = [0] * n_splits
    idx_test = [0] * n_splits
    i = 0
    current = 0
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        range_idx_train = np.arange(start, stop)
        idx_test[i] = np.array(total[start:stop])
        idx_train[i] = np.delete(total, range_idx_train)
        current = stop
        i = i + 1
        
    return idx_test, idx_train