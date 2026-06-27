"""Model factory for sequential recommendation models.

Registry:
    - sasrec: SASRec (Kang & McAuley 2018)
    - tisasrec: TiSASRec (Li et al. 2020)
    - tisasrec_cat: TiSASRec with category-conditioned bias
    - sasrec_rote: SASRec + RoTE time embeddings
    - tisasrec_rote: TiSASRec + RoTE time embeddings (with ablation support)
"""

import logging
from typing import Dict, Any

from .sasrec import SASRec, PointWiseFeedForward
from .tisasrec import TiSASRec, discretize_time_delta, get_time_buckets
from .tisasrec_cat import TiSASRecCat
from .rote import RoTEEncoder
from .sasrec_rote import SASRecRoTE
from .tisasrec_rote import TiSASRecRoTE

__all__ = [
    'SASRec',
    'TiSASRec',
    'TiSASRecCat',
    'SASRecRoTE',
    'TiSASRecRoTE',
    'RoTEEncoder',
    'PointWiseFeedForward',
    'build_model',
    'discretize_time_delta',
    'get_time_buckets',
]

logger = logging.getLogger(__name__)

_MODEL_REGISTRY = {
    'sasrec': SASRec,
    'tisasrec': TiSASRec,
    'tisasrec_cat': TiSASRecCat,
    'sasrec_rote': SASRecRoTE,
    'tisasrec_rote': TiSASRecRoTE,
}


def build_model(name: str, num_items: int, config: Dict[str, Any]):
    """Build a model by name with parameters from config.

    Args:
        name: Model name key (e.g. 'sasrec', 'tisasrec_rote').
        num_items: Number of items (excluding padding index 0).
        config: Full configuration dict (model section is read internally).

    Returns:
        An instance of the requested model.

    Raises:
        ValueError: If model name is unknown.
    """
    mc = config.get('model', {})

    if name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(_MODEL_REGISTRY.keys())}"
        )

    # Common parameters
    kwargs = {
        'num_items': num_items,
        'hidden_dim': mc.get('hidden_dim', 64),
        'num_layers': mc.get('num_layers', 2),
        'num_heads': mc.get('num_heads', 1),
        'dropout': mc.get('dropout', 0.2),
        'max_len': mc.get('max_len', 50),
    }

    # RoTE-specific parameters (for sasrec_rote and tisasrec_rote)
    if name in ('sasrec_rote', 'tisasrec_rote'):
        kwargs['rote_granularities'] = mc.get('rote_granularities', ['hour', 'day', 'week'])
        kwargs['rote_theta_base'] = mc.get('rote_theta_base', 10000.0)

    # TiSASRec base parameters (for tisasrec, tisasrec_cat, tisasrec_rote)
    if name in ('tisasrec', 'tisasrec_cat'):
        kwargs['time_bucket_defs'] = mc.get('time_bucket_defs', [0, 1, 6, 24, 168, 720])

    # TiSASRec-RoTE extra ablation parameters
    if name == 'tisasrec_rote':
        kwargs['time_bucket_defs'] = mc.get('time_bucket_defs', [0, 1, 6, 24, 168, 720])
        kwargs['use_relative_bias'] = mc.get('use_relative_bias', True)
        kwargs['use_rote'] = mc.get('use_rote', True)

    model_class = _MODEL_REGISTRY[name]
    logger.info(
        "Building model '%s' with hidden_dim=%d, num_layers=%d, max_len=%d",
        name, kwargs['hidden_dim'], kwargs['num_layers'], kwargs['max_len'],
    )
    return model_class(**kwargs)
