import json
import os
from pdb import set_trace

result_dir = "./gpt-4o"
result_file = os.path.join(result_dir, "result.json")
with open(result_file, 'r') as f:
    preds_all = json.load(f)
formatted_data = []
answer_file = "" #test json
with open(answer_file, 'r') as f:
    answers = json.load(f)
for index, (record,answer) in enumerate(zip(preds_all,answers)):
    try:
        human_message = record[0][0]
        gpt_response = record[0][1]
        gpt_response["value"] = gpt_response["value"]
        image_info = record[0][2]['image']
    except:
        human_message = record[0]
        gpt_response = record[1]
        gpt_response["value"] = gpt_response["value"]
        image_info = record[2]['image']
    
    dataset = image_info[0].split("/")[0]
   
    ans = answer["conversations"][1]["value"]
    formatted_record = {
        "conversations": [human_message, gpt_response],
        "image": image_info,
        "metadata": {
            "answer":ans,
            "dataset": dataset,
            "question_type": "close"
        }
    }
    
    
    formatted_record["conversations"] = [{"from": conv['from'], "value": conv['value']} for conv in formatted_record["conversations"]]
    
    formatted_data.append(formatted_record)

output_file_path = './gpt-4o/pred.json'
with open(output_file_path, 'w') as file:
    json.dump(formatted_data, file, indent=4)

print(f"Data has been successfully written to {output_file_path}")