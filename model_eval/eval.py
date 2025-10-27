import re
from rouge import Rouge
import argparse
import os
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

from pdb import set_trace



Pathology = ["catch,PAIP19,unipatho"]
class Eval:
    def __init__(self):
        self.periodStrip = re.compile("(?!<=\d)(\.)(?!\d)")
        self.commaStrip = re.compile("(\d)(\,)(\d)")
        self.punct = [
            ";",
            r"/",
            "[",
            "]",
            '"',
            "{",
            "}",
            "(",
            ")",
            "=",
            "+",
            "\\",
            "_",
            "-",
            ">",
            "<",
            "@",
            "`",
            ",",
            "?",
            "!",
        ]
        
    def processPunctuation(self, inText):
        outText = inText
        for p in self.punct:
            if (p + " " in inText or " " + p in inText) or (
                re.search(self.commaStrip, inText) != None
            ):
                outText = outText.replace(p, "")
            else:
                outText = outText.replace(p, " ")
        outText = self.periodStrip.sub("", outText, re.UNICODE)
        return outText
    
    def process(self, answer):
        answer = answer.replace("\n", " ")
        answer = answer.replace("\t", " ")
        answer = answer.strip()
        answer = self.processPunctuation(answer)
        answer = answer.strip('\'')
        answer = answer.strip('\"')
        answer = answer.strip(')')
        answer = answer.strip('(')
        answer = answer.strip().lower()
        match = re.search(r'answer:\s*([A-Za-z0-9]+)', answer)
        if match:
            return match.group(1)
        else:
            return answer

    def evaluate_rouge(self,preds):
        rouge = Rouge()
        acc = {'f': []}
        eval_list = []
        for i, res in enumerate(preds):
            sample_id = res['sample_id']
            # print(sample_id)
            gt_ans = self.process(res["gt_response"])
            pred_ans = self.process(res["pred_response"])
            # assert gt_ans != ''

            if gt_ans == '':
                continue
            
            if pred_ans == '':
                s = 0
            else:
                if len(pred_ans) > 512:
                    pred_ans = pred_ans[0: 512]
                s = rouge.get_scores(pred_ans, gt_ans)[0]['rouge-l']['f']
            acc['f'].append(s)
            eval_list.append({'score':str(round(s,3))})
        results = {'Rouge-L f': np.mean(acc['f'])}
        return results,eval_list


    def judge_multi_choice(self,sample):
        
        gt_ans = sample["metadata"]["answer"]
        pred_ans = sample["conversations"][1]["value"][0]

        if ":" in pred_ans:
            a_list = pred_ans.split(":")
            a_list = [a.strip() for a in a_list ]
            for a in a_list:
                if len(a) == 1 and a[-1] in ["a", "b", "c", "d", "e", "f", "g", "h"]:
                    pred_ans = a

        if pred_ans == gt_ans:
            return 1
        else:
            return 0

    def process_sample(self,sample):
        sample["metadata"]["answer"] = self.process(sample["metadata"]["answer"])
        sample["conversations"][1]["value"][0] = self.process(sample["conversations"][1]["value"][0])

    def evaluate_multichoice(self, preditions):
        correct = 0
        eval_list = []
        for i, sample in enumerate(preditions):
            self.process_sample(sample)
            score = self.judge_multi_choice(sample)
            
            sample['result'] = score
            eval_list.append({'score':str(score)})
            correct+=score
        return {'Accuracy':correct/len(preditions)},eval_list

    def evaluate_multi_choice_image(self,preditions):
        correct = 0
        eval_list = []
        for i,sample in enumerate(preditions):
            gt_ans = self.process(sample["metadata"]["answer"])
            pred_ans = self.process(sample["conversations"][1]["value"][0])
            

            if ":" in pred_ans:
                a_list = pred_ans.split(":")
                a_list = [a.strip() for a in a_list ]
                for a in a_list:
                    if len(a) == 1 and a[-1] in ["a", "b", "c", "d", "e", "f", "g", "h"]:
                        pred_ans = a

            if gt_ans == pred_ans:
                score = 1
            else:
                score = 0
            
            sample['result'] = score
            eval_list.append({'score':str(score)})
            correct+=score
        return {'Accuracy':correct/len(preditions)},eval_list

    def evaluate_acc(self, predictions):
        correct = 0
        eval_list = []
        for i, sample in enumerate(predictions):
            gt_ans = self.process(sample["metadata"]["answer"]) 
            pred_ans = self.process(sample["conversations"][1]["value"][0])  
            
            if ":" in pred_ans:
                a_list = pred_ans.split(":")
                a_list = [a.strip() for a in a_list]
                for a in a_list:
                    if len(a) == 1:
                        pred_ans = a

            if gt_ans == pred_ans:
                score = 1
            else:
                score = 0

            sample['result'] = score
            eval_list.append({'score': str(score)})
            correct += score

        accuracy = correct / len(predictions)
        return  accuracy, eval_list

    def evaluate_f1(self, predictions):
        y_true = []  
        y_pred = []  
        eval_list = []

        for i, sample in enumerate(predictions):

            gt_ans = self.process(sample["metadata"]["answer"])  
            try:
                pred_ans = self.process(sample["conversations"][1]["value"][0]) 
            except:
                set_trace()

            if ":" in pred_ans:
                a_list = pred_ans.split(":")
                a_list = [a.strip() for a in a_list]
                for a in a_list:
                    if len(a) == 1:
                        pred_ans = a

            # 收集真实值和预测值
            y_true.append(gt_ans)
            y_pred.append(pred_ans)

            # 判断是否正确
            if gt_ans == pred_ans:
                score = 1
            else:
                score = 0

            # 记录结果
            sample['result'] = score
            eval_list.append({'score': str(score)})

        # 计算 F1 分数
        f1 = f1_score(y_true, y_pred, average='weighted')  # 加权平均 F1 分数
        return  f1, eval_list

    def eval_confusion_matrix(self, predictions, save_path="confusion_matrix.png"):
        y_true = []  
        y_pred = []  

        for sample in predictions:
            gt_ans = self.process(sample["metadata"]["answer"])  
            pred_ans = self.process(sample["conversations"][1]["value"][0])  

     
            if ":" in pred_ans:
                a_list = pred_ans.split(":")
                a_list = [a.strip() for a in a_list]
                for a in a_list:
                    if len(a) == 1 :
                        pred_ans = a


            y_true.append(gt_ans)
            y_pred.append(pred_ans)
        labels = np.unique(np.concatenate((y_true, y_pred)))

        conf_matrix = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels)
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title("Confusion Matrix")
        plt.tight_layout()

        plt.savefig(save_path, dpi=300, bbox_inches='tight')  

       
if __name__ == "__main__":

    result_dir = "./gpt-4o"
    result_file = os.path.join(result_dir, "pred.json")

    if not os.path.exists(result_file):
        print('No prediction file found')
        exit(0)
    with open(result_file, 'r') as f:
        preds_all = json.load(f)
    preds_all_dict = dict()
    for pred in preds_all:
        if pred["metadata"]["dataset"] not in preds_all_dict:
            preds_all_dict[pred["metadata"]["dataset"]] = list()
        preds_all_dict[pred["metadata"]["dataset"]].append(pred)

    E = Eval()

    eval_result_list = dict()
    eval_result_list_detail = dict()
    for dataset in preds_all_dict:
        
        preds = preds_all_dict[dataset]
        # set_trace()
        question_type = preds[0]["metadata"]["question_type"]

        if question_type == 'open-ended':
            if dataset == 'Pathology':
                eval_result, eval_list = E.evaluate_rouge(preds)
            else :
                f1_result, f1_eval_list = E.evaluate_f1(preds)
                acc_result, acc_eval_list = E.evaluate_acc(preds)

                eval_result = {
                    "f1": f1_result,
                    "acc": acc_result
                }
                eval_list = {
                    "F1": f1_eval_list,
                    "ACC": acc_eval_list
                }
                path = os.path.join(result_dir,f'confusion_matrix_open{dataset}.png')
                E.eval_confusion_matrix(preds,path)
        elif question_type == 'close':
            f1_result, f1_eval_list = E.evaluate_f1(preds)
            acc_result, acc_eval_list = E.evaluate_acc(preds)

            eval_result = {
                "f1": f1_result,
                "acc": acc_result
            }
            eval_list = {
                "F1": f1_eval_list,
                "ACC": acc_eval_list
            }
            # path = os.path.join(result_dir,f'confusion_matrix_close{dataset}.png')
            # E.eval_confusion_matrix(preds,path)
                        

        else:
            eval_result = 'Dataset not supported'
            print('Dataset not supported')
            exit(0)

        eval_result_list[dataset] = eval_result
        eval_result_list_detail[dataset] = eval_list

    os.makedirs(result_dir, exist_ok=True)
    with open(os.path.join(result_dir, 'eval_dataset.json'), 'w') as f:
        json.dump(eval_result_list, f, indent=4)

    with open(os.path.join(result_dir,'eval_dataset_details.json'), 'w') as f:
        json.dump(eval_result_list_detail, f, indent=4)


    eval_cat_list = dict()
    print()

    # PAIP19
    score = 0
    count = 0
    score1 = 0
    for dataset in eval_result_list:
        if dataset in "PAIP19":
            count += 1
            score += list(eval_result_list[dataset].values())[0]
            score1 += list(eval_result_list[dataset].values())[1]
    if count > 0:
        score /= count
        score1 /= count
        eval_score = {
            "f1": score,
            "acc": score1
        }
        eval_cat_list["PAIP19"] = eval_score
        print("PAIP19_acc", end = ':  ')
        print('{:.2f}'.format(100 * score1))
        print("PAIP19_f1", end = ':  ')
        print('{:.2f}'.format(100 * score))
    # catch
    score = 0
    count = 0
    score1 = 0
    for dataset in eval_result_list:
        if dataset in "catch":
            count += 1
            score += list(eval_result_list[dataset].values())[0]
            score1 += list(eval_result_list[dataset].values())[1]
    if count > 0:

        score /= count
        score1 /= count
        eval_score = {
            "f1": score,
            "acc": score1
        }
        eval_cat_list["catch"] = eval_score
        print("-"*20)
        print("catch_acc", end = ':  ')
        print('{:.2f}'.format(100 * score1))
        print("catch_f1", end = ':  ')
        print('{:.2f}'.format(100 * score))
    # unipatho
    score = 0
    count = 0
    score1 = 0
    for dataset in eval_result_list:
        if dataset in "unipatho":
            count += 1
            score += list(eval_result_list[dataset].values())[0]
            score1 += list(eval_result_list[dataset].values())[1]
    if count > 0:
        score /= count
        score1 /= count
        eval_score = {
            "f1": score,
            "acc": score1
        }
        eval_cat_list["unipatho"] = eval_score
        print("\nunipatho_acc", end = ':  ')
        print('{:.2f}'.format(100 * score1))
        print("unipatho_f1", end = ':  ')
        print('{:.2f}'.format(100 * score))

    with open(os.path.join(result_dir,'eval_cat.json'), 'w') as f:
        json.dump(eval_cat_list, f, indent=4)