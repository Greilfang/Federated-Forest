from sklearn.datasets.samples_generator import make_blobs

def load_random(n_samples=1000, centers=3, n_features=6,test_rate=0.2):
    X, y = make_blobs(n_samples=n_samples, centers = centers, n_features=n_features)
    edge = int(n_samples * (1-test_rate))
    train_set={
        'data':X[:edge],
        'target':y[:edge],
        'class_num':centers
    }
    test_set={
        'data':X[edge:],
        'target':y[edge:],
        'class_num':centers
    }
    return train_set,test_set

