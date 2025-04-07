import json
import numpy as np
import sys

out_file = sys.argv[1]
qa_reward_path = sys.argv[2]
qr_result = json.load(open(sys.argv[3]))

qr_result_dic = {}
for item in qr_result:
    id = str(item['Conversation_no'])+'_'+str(item['Turn_no'])
    qr_result_dic[id] = item
qa_reward = json.load(open(qa_reward_path))

tau = 1
topk = 5

def softmax(s, tau=1.0):
    s = np.array(s)
    return np.exp(s/tau)/sum(np.exp(s/tau))

def postprocess_qa_single_psg(qa_reward, invalid_keys=None, weighted_sum=False, sum_loss=False,topk=5, tau=1.0):
    pseudo_qr_reward = {}

    data_idx = 0
    for q_h, q_g, q_e in zip(qa_reward['rewrite1'], qa_reward['rewrite2'], qa_reward['rewrite3']):
        
        idx = q_h['sample']['id']

        pseudo_qr_reward[idx] = {}
        rewrites = [q_h['scores'][0] if q_h['scores'] else None, \
                q_g['scores'][0] if q_g['scores'] else None, \
                q_e['scores'][0] if q_e['scores'] else None]
        if weighted_sum:
            if q_h['scores']:
                q_h_re = [i[1] for i in q_h['sample']['doc_labels']][:topk]
                q_h_re = softmax(q_h_re, tau=tau)
                q_h_qa = q_h['scores'][:topk]
                q_h_loss = np.dot(q_h_re, q_h_qa)
                rewrites[0] = q_h_loss
            if q_g['scores']:
                q_g_re = [i[1] for i in q_g['sample']['doc_labels']][:topk]
                q_g_re = softmax(q_g_re, tau=tau)
                q_g_qa = q_g['scores'][:topk]
                q_g_loss = np.dot(q_g_re, q_g_qa)
                rewrites[1] = q_g_loss
            if q_e['scores']:
                q_e_re = [i[1] for i in q_e['sample']['doc_labels']][:topk]
                q_e_re = softmax(q_e_re, tau=tau)
                q_e_qa = q_e['scores'][:topk]
                q_e_loss = np.dot(q_e_re, q_e_qa)
                rewrites[2] = q_e_loss
        elif sum_loss:
            if q_h['scores']:
                q_h_loss = sum(q_h['scores'][:topk])
                rewrites[0] = q_h_loss
            if q_g['scores']:
                q_g_loss = sum(q_g['scores'][:topk])
                rewrites[1] = q_g_loss
            if q_e['scores']:
                q_e_loss = sum(q_e['scores'][:topk])
                rewrites[2] = q_e_loss
        
        
        qa_reward['rewrite1'][data_idx]['scores_per_psg'] = qa_reward['rewrite1'][data_idx]['scores']
        qa_reward['rewrite2'][data_idx]['scores_per_psg'] = qa_reward['rewrite2'][data_idx]['scores']
        qa_reward['rewrite3'][data_idx]['scores_per_psg'] = qa_reward['rewrite3'][data_idx]['scores']

        qa_reward['rewrite1'][data_idx]['scores'] = [rewrites[0]]
        qa_reward['rewrite2'][data_idx]['scores'] = [rewrites[1]]
        qa_reward['rewrite3'][data_idx]['scores'] = [rewrites[2]]

        validity = True
        if invalid_keys: 
            if idx in invalid_keys: validity = False
        
        qa_reward['rewrite1'][data_idx]['validity'] = validity
        qa_reward['rewrite2'][data_idx]['validity'] = validity
        qa_reward['rewrite3'][data_idx]['validity'] = validity
        
        data_idx += 1
            
        
    return qa_reward


qa_reward_postprocess = postprocess_qa_single_psg(qa_reward, weighted_sum=True, topk=topk, tau=tau)



combine_data = []
nopsg, invalid = 0, 0
for qa_human, qa_gpt, qa_editor in zip(qa_reward_postprocess['rewrite1'], qa_reward_postprocess['rewrite2'], qa_reward_postprocess['rewrite3']):
    
    assert qa_human['sample']['id'] == qa_gpt['sample']['id'] == qa_editor['sample']['id'] 
    
    qr = qr_result_dic[qa_human['sample']['id']]


    if not qa_human['validity']:invalid += 1
    combine_data.append({
        'Conversation_no': qr['Conversation_no'], 
        'Turn_no': qr['Turn_no'], 
        'Question': qr['Question'], 
        'NewContext': qr['NewContext'], 
        'Truth_answer': qr['Truth_answer'], 
        # 'Truth_passages': qr['Truth_passages'],
        # 'Conversation_source': qr['Conversation_source'], 
        'rewrites': {
            'rewrite1': {
                'question': qa_human['sample']['question'],
                'score':  qa_human['scores'][0],
                'validity': qa_human['validity']
            },
            'rewrite2': {
                'question': qa_gpt['sample']['question'],
                'score':  qa_gpt['scores'][0],
                'validity': qa_gpt['validity']   
            },
            'rewrite3': {
                'question': qa_editor['sample']['question'],
                'score':  qa_editor['scores'][0],
                'validity': qa_editor['validity']
            }
        }
    })
with open(out_file, 'w') as f:
    json_ = json.dumps(combine_data, indent=1)
    f.write(json_)