import numpy as np
from sklearn.model_selection import ShuffleSplit
from keras.callbacks import EarlyStopping


def shuffle_weights(model, weights=None):
    """Randomly permute the weights in `model`, or the given `weights`.
    This is a fast approximation of re-initializing the weights of a model.
    Assumes weights are distributed independently of the dimensions of the weight tensors
      (i.e., the weights have the same distribution along each dimension).
    :param Model model: Modify the weights of the given model.
    :param list(ndarray) weights: The model's weights will be replaced by a random permutation of these weights.
      If `None`, permute the model's current weights.
    """
    if weights is None:
        weights = model.get_weights()
    weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
    # Faster, but less random: only permutes along the first dimension
    # weights = [np.random.permutation(w) for w in weights]
    model.set_weights(weights)


def validate(model, X, y, nb_epoch=25, batch_size=128,
             stop_early=True, folds=10, test_size=None, shuffle=True, verbose=True):
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='auto')

    total_score = []
    if test_size is None:
        if folds == 1:
            test_size = 0.25
        else:
            test_size = 1 - (1. / folds)
    kf = ShuffleSplit(n_splits=folds, test_size=test_size)
    for fold, (train_index, test_index) in enumerate(kf.split(X, y)):
        shuffle_weights(model)
        if fold > 0:
            print("FOLD:", fold)
            print("-" * 40)
            model.reset_states()
            callbacks = [early_stopping] if True else None
        hist = model.fit(X[train_index], y[train_index], batch_size=batch_size, shuffle=shuffle,
                         validation_data=(X[test_index], y[test_index]),
                         callbacks=[early_stopping], verbose=verbose)
        total_score.append(hist.history["val_acc"][-1])
    return np.mean(total_score)


def run(model, X, y, nb_epoch=25, batch_size=128, stop_early=True, test_size=None, shuffle=True,
        verbose=True):
    return validate(model, X, y, nb_epoch, batch_size, stop_early, 1, test_size, shuffle, verbose)
