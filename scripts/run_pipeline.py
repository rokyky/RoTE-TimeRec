# Run full recommendation pipeline end-to-end.

import argparse, logging, os, sys, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.utils.config import load_config
from src.pipeline.base import CandidateList, Candidate
from src.pipeline.runner import PipelineRunner
from src.pipeline.recall import PopularityRecall
from src.pipeline.pre_rank import SimplePreRank
from src.pipeline.rank import SequenceRanker
from src.pipeline.re_rank import MMRReRank
from src.models.sasrec import SASRec

def generate_synthetic_data(num_users=50, num_items=30, max_len=15):
    random.seed(42)
    sequences = {}
    for uid in range(num_users):
        length = random.randint(3, max_len)
        seq = [random.randint(1, num_items) for _ in range(length)]
        sequences[uid] = seq
    return sequences, num_items

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    logging.basicConfig(level=logging.INFO)

    # Data
    sequences, num_items = generate_synthetic_data()
    split = int(len(sequences) * 0.8)
    users = list(sequences.keys())
    train_seq = {u: sequences[u] for u in users[:split]}
    eval_seq = {u: sequences[u] for u in users[split:]}

    # Init candidates: all users need candidates
    candidates = CandidateList()
    for uid in eval_seq:
        candidates.add(uid, Candidate(uid, 0, 0.0, {}, "init"))

    # Context
    context = {"sequences": train_seq, "user_history": train_seq}

    # Pipeline
    runner = PipelineRunner(config)
    pop = PopularityRecall(top_k=30)
    item_counts = {}
    for seq in train_seq.values():
        for item in seq:
            item_counts[item] = item_counts.get(item, 0) + 1
    pop.fit(item_counts)
    runner.add_stage("recall", pop)

    # Only add rank if we have a model
    model = SASRec(num_items, hidden_dim=16, max_len=15)
    ranker = SequenceRanker(model, keep_k=15)
    runner.add_stage("rank", ranker)

    rerank = MMRReRank(keep_k=10, lam=0.5)
    runner.add_stage("rerank", rerank)

    result = runner.run(candidates, context)
    total = len(result)
    users_with_cands = sum(1 for uid in eval_seq if len(result.get(uid)) > 0)
    print(f"Pipeline complete: {total} candidates for {users_with_cands}/{len(eval_seq)} users")
    return result

if __name__ == "__main__": main()