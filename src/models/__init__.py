"""序列推荐模型的工厂函数。

注册表：
    - sasrec: SASRec（Kang & McAuley 2018）
    - tisasrec: TiSASRec（Li et al. 2020）
    - tisasrec_cat: 带类别条件化偏置的 TiSASRec
    - sasrec_rote: SASRec + RoTE 时间嵌入
    - tisasrec_rote: TiSASRec + RoTE 时间嵌入（带消融支持）
    - dssm: 双塔向量召回模型
"""

import logging
from typing import Dict, Any

from .sasrec import SASRec, PointWiseFeedForward
from .tisasrec import TiSASRec, discretize_time_delta, get_time_buckets
from .tisasrec_cat import TiSASRecCat
from .rote import RoTEEncoder
from .sasrec_rote import SASRecRoTE
from .tisasrec_rote import TiSASRecRoTE
from .dssm import DSSM, DSSMDataset, collate_dssm, train_epoch_softmax, train_epoch_bpr

__all__ = [
    'SASRec',
    'TiSASRec',
    'TiSASRecCat',
    'SASRecRoTE',
    'TiSASRecRoTE',
    'DSSM',
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
    'dssm': DSSM,
}


def build_model(name: str, num_items: int, config: Dict[str, Any]):
    """通过名称和配置参数构建模型。

    参数：
        name: 模型名称键（如 'sasrec', 'tisasrec_rote', 'dssm'）。
        num_items: 物品数量（不含填充索引 0）。
        config: 完整配置字典（内部读取 model 部分）。

    返回：
        所请求模型的实例。

    抛出：
        ValueError: 如果模型名称未知。
    """
    mc = config.get('model', {})
    dc = config.get('dssm', {})

    if name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(_MODEL_REGISTRY.keys())}"
        )

    # 通用参数
    kwargs = {
        'num_items': num_items,
        'hidden_dim': mc.get('hidden_dim', 64),
        'dropout': mc.get('dropout', 0.2),
    }

    if name == 'dssm':
        kwargs['num_users'] = dc.get('num_users', 0)
        kwargs['mlp_dims'] = dc.get('mlp_dims', None)

    # 序列模型参数
    if name != 'dssm':
        kwargs.update({
            'num_layers': mc.get('num_layers', 2),
            'num_heads': mc.get('num_heads', 1),
            'max_len': mc.get('max_len', 50),
        })

    # RoTE 特定参数（用于 sasrec_rote 和 tisasrec_rote）
    if name in ('sasrec_rote', 'tisasrec_rote'):
        kwargs['rote_granularities'] = mc.get('rote_granularities', ['hour', 'day', 'week'])
        kwargs['rote_theta_base'] = mc.get('rote_theta_base', 10000.0)

    # TiSASRec 基础参数（用于 tisasrec, tisasrec_cat, tisasrec_rote）
    if name in ('tisasrec', 'tisasrec_cat'):
        kwargs['time_bucket_defs'] = mc.get('time_bucket_defs', [0, 1, 6, 24, 168, 720])

    # TiSASRec-RoTE 额外消融参数
    if name == 'tisasrec_rote':
        kwargs['time_bucket_defs'] = mc.get('time_bucket_defs', [0, 1, 6, 24, 168, 720])
        kwargs['use_relative_bias'] = mc.get('use_relative_bias', True)
        kwargs['use_rote'] = mc.get('use_rote', True)

    model_class = _MODEL_REGISTRY[name]
    logger.info(
        "Building model '%s' with hidden_dim=%d",
        name, kwargs['hidden_dim'],
    )
    return model_class(**kwargs)
