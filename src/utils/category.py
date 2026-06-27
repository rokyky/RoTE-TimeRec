import numpy as np
import pandas as pd
from collections import Counter


def compute_category_stats(train_df, item2cat, user_col='reviewerID',
                           item_col='asin', time_col='unixReviewTime'):
    """计算分类分布统计，用于 Day 0 完整性检查。
    
    参数：
        train_df: 包含用户-商品-时间交互的 DataFrame
        item2cat: 商品 asin 到分类字符串的映射字典
    """
    stats = {}

    # 覆盖率：具有叶子分类的商品比例
    items_with_cat = sum(1 for item in train_df[item_col].unique() if item in item2cat)
    total_items = train_df[item_col].nunique()
    stats['item_coverage'] = items_with_cat / total_items

    # 用户级覆盖率：具有分类的交互比例
    train_with_cat = train_df.copy()
    train_with_cat['has_cat'] = train_with_cat[item_col].isin(item2cat)
    stats['interaction_coverage'] = train_with_cat['has_cat'].mean()

    # 每个用户的唯一叶子分类数
    user_cats = train_df.groupby(user_col)[item_col].apply(
        lambda items: set(item2cat.get(i) for i in items if i in item2cat)
    )
    user_cat_counts = user_cats.apply(len)
    stats['avg_unique_categories_per_user'] = user_cat_counts.mean()
    stats['median_unique_categories_per_user'] = user_cat_counts.median()
    stats['std_unique_categories_per_user'] = user_cat_counts.std()

    # 每个用户的分类熵
    def cat_entropy(items):
        cats = [item2cat.get(i) for i in items if i in item2cat]
        if not cats:
            return 0.0
        counts = Counter(cats)
        total = sum(counts.values())
        probs = [c / total for c in counts.values()]
        return -sum(p * np.log(p) for p in probs)

    user_entropies = train_df.groupby(user_col)[item_col].apply(cat_entropy)
    stats['avg_user_category_entropy'] = user_entropies.mean()
    stats['std_user_category_entropy'] = user_entropies.std()

    # 跨分类转换率（用户序列中的相邻对）
    def cross_cat_rate(group):
        group = group.sort_values(time_col)
        items = group[item_col].tolist()
        if len(items) < 2:
            return 0.0
        cross = 0
        total = 0
        for i in range(len(items) - 1):
            cat1 = item2cat.get(items[i])
            cat2 = item2cat.get(items[i + 1])
            if cat1 is not None and cat2 is not None:
                total += 1
                if cat1 != cat2:
                    cross += 1
        return cross / total if total > 0 else 0.0

    cross_rates = train_df.groupby(user_col).apply(
        lambda g: cross_cat_rate(g)
    )
    stats['avg_cross_category_transition_rate'] = cross_rates.mean()
    stats['std_cross_category_transition_rate'] = cross_rates.std()

    # same_cat=1 比例（相邻对）
    def same_cat_proportion(group):
        group = group.sort_values(time_col)
        items = group[item_col].tolist()
        if len(items) < 2:
            return 0.0
        same = 0
        total = 0
        for i in range(len(items) - 1):
            cat1 = item2cat.get(items[i])
            cat2 = item2cat.get(items[i + 1])
            if cat1 is not None and cat2 is not None:
                total += 1
                if cat1 == cat2:
                    same += 1
        return same / total if total > 0 else 0.0

    same_cat_rates = train_df.groupby(user_col).apply(
        lambda g: same_cat_proportion(g)
    )
    stats['avg_same_category_pair_ratio'] = same_cat_rates.mean()
    stats['std_same_category_pair_ratio'] = same_cat_rates.std()

    # 热门分类
    all_cats = []
    for item in train_df[item_col].unique():
        if item in item2cat:
            all_cats.append(item2cat[item])
    cat_counts = Counter(all_cats)
    stats['num_unique_categories'] = len(cat_counts)
    stats['top_10_categories'] = cat_counts.most_common(10)

    return stats


def print_category_report(dataset_name, stats):
    """打印格式化的分类完整性报告。"""
    print(f"\n{'='*60}")
    print(f"  Category Sanity Check: {dataset_name}")
    print(f"{'='*60}")
    print(f"  Item coverage (with cat):     {stats['item_coverage']:.2%}  "
          f"(threshold: >= 90%)")
    print(f"  Interaction coverage:           {stats['interaction_coverage']:.2%}")
    print(f"  Avg unique cats per user:       {stats['avg_unique_categories_per_user']:.2f}  "
          f"(Sports > Beauty expected)")
    print(f"  Median unique cats per user:    {stats['median_unique_categories_per_user']:.2f}")
    print(f"  Avg user cat entropy:           {stats['avg_user_category_entropy']:.3f}  "
          f"(Sports > Beauty expected)")
    print(f"  Cross-cat transition rate:      {stats['avg_cross_category_transition_rate']:.2%}  "
          f"(Sports - Beauty >= 10% expected)")
    print(f"  Same-cat adjacent pair ratio:   {stats['avg_same_category_pair_ratio']:.2%}  "
          f"(should be in [0.1, 0.9])")
    print(f"  Num unique categories:          {stats['num_unique_categories']}")
    print(f"  Top 10 categories:")
    for cat, count in stats['top_10_categories'][:10]:
        print(f"    - {cat}: {count}")
    print(f"{'='*60}\n")
