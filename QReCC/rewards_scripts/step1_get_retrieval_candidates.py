import os
import json
import argparse
from tqdm import tqdm

from shared_utils.indexing_utils import SparseIndexer, DocumentCollection
from shared_utils import get_logger


logger = get_logger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default="qrecc")
    parser.add_argument('--split', type=str, default="train")
    parser.add_argument('--raw_data_path', type=str, default='/mnt/file-201-project-disk-m/project/dialogue/AI4Future/research-projects/conv-query-rewrite')
    parser.add_argument('--source_data_path', type=str, default='/mnt/file-201-project-disk-m/project/dialogue/AI4Future/research-projects/conv-query-rewrite/InfoCQR/get_reward/candidates/train-sampled10k_chatgpt_ICL_editor.json')
    parser.add_argument('--preprocessed_data_path', type=str, default='./outputs/retrieval/bm25/')
    parser.add_argument('--pyserini_index_path', type=str, default='/mnt/file-201-project-disk-m/project/dialogue/AI4Future/research-projects/conv-query-rewrite/InfoCQR/datasets-tianhua/preprocessed/qrecc/pyserini_index')
    parser.add_argument('--data_file', type=str, default='rewrite_sampling_from_best_sft.json')
    parser.add_argument('--top_k', type=int, default=5)
    args = parser.parse_args()
    
    os.makedirs(args.preprocessed_data_path, exist_ok=True)

    k_1 = 0.82
    b = 0.68

    indexer = SparseIndexer(args.pyserini_index_path)
    indexer.set_retriever(k_1, b)


    source_data = json.load(open(args.source_data_path, "r", encoding="utf-8")) 
    source_data_dic = {}
    for item in source_data:
        id = str(item['Conversation_no'])+'_'+str(item['Turn_no'])
        assert id not in source_data_dic
        source_data_dic[id] = item

    data = []   
    with open(f"{args.raw_data_path}/{args.data_file}", "r", encoding="utf-8") as f:
        for line in f: data.append(json.loads(line))
    out_sfx = args.data_file.strip(".json")
    output_file = os.path.join(args.preprocessed_data_path, f"{out_sfx}.json")
    print(output_file)


    results = []
    for each in tqdm(data):
        id = str(each['Conversation_no'])+'_'+str(each['Turn_no'])
        assert id in source_data_dic
        item = source_data_dic[id]

        if not item['Truth_answer']: continue


        result = {
            'Conversation_no': item['Conversation_no'],
            'Turn_no': item['Turn_no'],
            # 'Conversation_source': item['Conversation_source'],
            'Question': item['Question'],
            'Truth_answer': item['Truth_answer'],
            # 'Truth_passages': item['Truth_passages'],
            'NewContext': item['NewContext'],
            'rewrite': {'rewrite1': {'rewrite': each['rewrite'][0].strip(), 'docs': []}, 
                        'rewrite2':  {'rewrite': each['rewrite'][1].strip(), 'docs': []},
                        'rewrite3': {'rewrite': each['rewrite'][2].strip(), 'docs': []}}
        }
        for q_type in ['rewrite1', 'rewrite2', 'rewrite3']:
            q = result['rewrite'][q_type]['rewrite']
            if not q: continue # empty rewrite candidate
            result['rewrite'][q_type]['docs'] = indexer.retrieve(q, args.top_k)
        results.append(result)

        if len(results) % 1000 == 1:
            json.dump(
                results,
                open(output_file, "w"),
                indent=1
            )
    json.dump(
        results,
        open(output_file, "w"),
        indent=1
    )


