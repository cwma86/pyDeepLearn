"""pyDeepLearn: a from-scratch, NumPy-based deep learning framework.

``pyDeepLearn`` was written to explore the inner workings of neural network
training -- forward and backward propagation, layer composition, and objective
(loss) functions -- without relying on an existing deep learning library.

The package is organized into three groups of modules:

* Layers -- :mod:`~pyDeepLearn.InputLayer`,
  :mod:`~pyDeepLearn.FullyConnectedLayer`, :mod:`~pyDeepLearn.LinearLayer`,
  :mod:`~pyDeepLearn.ReLuLayer`, :mod:`~pyDeepLearn.SigmoidLayer`,
  :mod:`~pyDeepLearn.SoftmaxLayer`, and :mod:`~pyDeepLearn.TanhLayer`.
* Objective functions -- :mod:`~pyDeepLearn.LeastSquares`,
  :mod:`~pyDeepLearn.LogLoss`, and :mod:`~pyDeepLearn.CrossEntropy`.
* Training utilities -- :mod:`~pyDeepLearn.RunUtilities`.

Typical usage::

    import numpy as np

    from pyDeepLearn.InputLayer import InputLayer
    from pyDeepLearn.FullyConnectedLayer import FullyConnectedLayer
    from pyDeepLearn.LinearLayer import LinearLayer
    from pyDeepLearn.LeastSquares import LeastSquares
    from pyDeepLearn.RunUtilities import run_layers

    X_train = np.random.random((100, 4))
    Y_train = np.random.random((100, 2))

    layers = [
        InputLayer(X_train),
        FullyConnectedLayer(X_train.shape[1], 16),
        LinearLayer(),
        FullyConnectedLayer(16, Y_train.shape[1]),
        LinearLayer(),
        LeastSquares(),
    ]
    run_layers(layers, X_train, Y_train, max_epoch=100)
"""
