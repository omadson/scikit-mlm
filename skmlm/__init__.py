"""skmlm implements MLM models."""

from .mlm import MLM, MLMC, NN_MLM, ON_MLM, w_MLM, OS_MLM, FCM_MLM, L12_MLM, L2_MLM, C_MLM, OS_MLMR, ELM, OPELM
from .utils import load_dataset, get_metrics, get_metrics_MLM_gs, get_metrics_KNN

__all__ = ['MLM', 'MLMC', 'NN_MLM', 'ON_MLM', 'w_MLM', 'OS_MLM', 'FCM_MLM', 'L12_MLM', 'L2_MLM', 'C_MLM', 'OS_MLMR', 'ELM']

__version__ = '0.1.1'