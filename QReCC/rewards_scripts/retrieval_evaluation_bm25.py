import os
import json
import argparse
from tqdm import tqdm

from shared_utils.indexing_utils import SparseIndexer, DocumentCollection
from shared_utils import get_logger


logger = get_logger(__name__)


def read_qrecc_data(dataset, read_by="all", is_test=False):
    examples = []
    for data in tqdm(dataset):
        guid = f"{data['Conversation_no']}_{data['Turn_no']}"
        examples.append([guid, data['rewrite']])
        
    return examples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default="qrecc")
    parser.add_argument('--split', type=str, default="dev")
    parser.add_argument('--read_by', type=str, default="all")
    parser.add_argument('--data_file', type=str, default=None)
    parser.add_argument('--preprocessed_data_path', type=str, default=None)
    parser.add_argument('--pyserini_index_path', type=str, default=None)
    parser.add_argument('--out_sfx', type=str, default=None)
    parser.add_argument('--qrel_path', type=str, default=None)
    parser.add_argument('--top_k', type=int, default=100)
    args = parser.parse_args()
    
    os.makedirs(args.preprocessed_data_path, exist_ok=True)
    out_sfx = args.data_file.rsplit('/')[-1].strip(".json") if not args.out_sfx else args.out_sfx
    print(out_sfx)

    k_1 = 0.82
    b = 0.68

    indexer = SparseIndexer(args.pyserini_index_path)
    indexer.set_retriever(k_1, b)

    qrels = json.load(open(os.path.join(args.qrel_path, f"qrels_{args.split}.txt"), "r"))


    data = []
    with open(args.data_file) as f:
        for line in f: data.append(json.loads(line))

    raw_examples = read_qrecc_data(data)

    print(f'Total number of queries: {len(raw_examples)}')

    scores = {}

    for line in tqdm(raw_examples):
        qid, q = line

        no_rels = False
        if args.split == "test" or args.split == "dev":
            if list(qrels[qid].keys())[0] == '':
                no_rels = True
        if no_rels:
            continue
        
        if not q:
            scores[qid] = {}
            continue

        retrieved_passages = indexer.retrieve(q, args.top_k)
        score = {}
        for passage in retrieved_passages:
            score[passage["id"]] = passage["score"]
        scores[qid] = score

        if len(scores) % 1000 == 1:
            json.dump(
                scores,
                open(os.path.join(args.preprocessed_data_path, f"{args.split}_{args.read_by}_{out_sfx}_bm25.json"), "w"),
                indent=1
            )

    
    
    json.dump(
        scores,
        open(os.path.join(args.preprocessed_data_path, f"{args.split}_{args.read_by}_{out_sfx}_bm25.json"), "w"),
        indent=1
    )
    
