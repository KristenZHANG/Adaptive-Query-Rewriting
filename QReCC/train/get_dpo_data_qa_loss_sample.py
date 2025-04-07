import sys
import json
print(sys.argv[1])
print(sys.argv[2])

source_data_path = sys.argv[1]
with open(source_data_path) as f:
    source_data = json.load(f)

processed_data = []
for i, item in enumerate(source_data):
    question = item["Question"].strip()
    question = question if question.endswith("?") else question + "?"
    context = item["NewContext"]
    response = item["Truth_answer"].strip()
    response = response if response.endswith(".") else response + "."

    if len(context) ==0:
        continue
    if item['rewrites']["rewrite1"]["validity"]==False:
        continue

    assert len(context)%2 == 0
    collapsed_context = ""

    for j, utterance in enumerate(context):
        if j%2==0:
            utterance = utterance if utterance.endswith("?") else utterance + "?"
            collapsed_context += "Q: " + utterance + " "
        else:
            utterance = utterance if utterance.endswith(".") else utterance + "."
            collapsed_context += "A: " + utterance if j==len(context)-1 else "A: " + utterance + "\n"

    labels = []
    if len(item['rewrites']["rewrite1"]["question"])>1:
        labels.append((item['rewrites']["rewrite1"]["question"], item['rewrites']["rewrite1"]["score"]))
    if len(item['rewrites']["rewrite2"]["question"])>1:
        labels.append((item['rewrites']["rewrite2"]["question"], item['rewrites']["rewrite2"]["score"]))
    if len(item['rewrites']["rewrite3"]["question"])>1:
        labels.append((item['rewrites']["rewrite3"]["question"], item['rewrites']["rewrite3"]["score"]))
    # import pdb; pdb.set_trace()
    # if len(labels)<3:
    #     import pdb; pdb.set_trace()
    labels.sort(key=lambda x: x[1])
    preference_pairs = []
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
            if abs(labels[i][1]-labels[j][1])<0.001:
                continue
            preference_pairs.append([labels[i][0], labels[j][0]])

    for pair in preference_pairs:
        cho = pair[0] if pair[0].endswith("?") else pair[0] + "?"
        rej = pair[1] if pair[1].endswith("?") else pair[1] + "?"

        processed_data.append(json.dumps({"conversation": collapsed_context, "question": question, "chosen": cho, "rejected": rej}) + "\n")
    # import pdb; pdb.set_trace()
print(len(processed_data))

with open(sys.argv[2], "w") as f:
    f.writelines(processed_data)