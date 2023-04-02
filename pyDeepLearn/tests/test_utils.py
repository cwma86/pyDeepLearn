import numpy as np
import logging


def np_array_comp(test_data, truth_data):
  if np.allclose(test_data, truth_data, atol=1e-04):
    return True
  else:
    logging.warning(f"test_data  doesnt match truth_data")
    logging.warning(f"test_data: {test_data}")
    logging.warning(f"truth_data: {truth_data}")
    return False
