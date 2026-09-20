pyDeepLearn documentation
=========================

``pyDeepLearn`` is a from-scratch, NumPy-based deep learning framework. It was
written to explore the inner workings of neural network training -- forward and
backward propagation, layer composition, and objective (loss) functions --
without relying on an existing autograd or deep learning library.

The package is built around two abstract base types:

* :class:`pyDeepLearn.LayerInterface.Layer` -- the interface implemented by
  every layer (input, fully connected, recurrent, and activation layers).
* :class:`pyDeepLearn.objectiveFuncInterface.objectiveFuncInterface` -- the
  interface implemented by every objective (loss) function.

Layers are composed into an ordered list and trained with the helpers in
:mod:`pyDeepLearn.RunUtilities`.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
