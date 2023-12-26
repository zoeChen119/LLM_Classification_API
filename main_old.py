import os
from http import HTTPStatus
import dashscope
import pandas as pd
from tqdm import tqdm
import time
import json
import requests
from spark_v3 import spark_main
from metrics import text_classification_metrics

import warnings
warnings.filterwarnings('ignore')

def mkdir(path):
 
	folder = os.path.exists(path)
 
	if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
		os.makedirs(path)            #makedirs 创建文件时如果路径不存在会创建这个路径
		print("---  new folder...  ---")
		print(path)
		print("---  OK  ---")
 
	else:
		print("---  There is this folder!  ---")
		print(path)
	return path

class Model2FuncClass(object):      
    def get_access_token(self, api_key, secret_key):
        """
        使用 API Key，Secret Key 获取access_token，替换下列示例中的应用API Key、应用Secret Key
        """
            
        url = f"https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={api_key}&client_secret={secret_key}"
        
        payload = json.dumps("")
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        
        response = requests.request("POST", url, headers=headers, data=payload)
        return response.json().get("access_token")
    
    def qwen_API(self, prompt, api_key='sk-419d5b7d8b5c4ebd8140b499e870329c'):
        dashscope.api_key = api_key

        response = dashscope.Generation.call(
            model=dashscope.Generation.Models.qwen_max,
            prompt=prompt
        )
        # The response status_code is HTTPStatus.OK indicate success,
        # otherwise indicate request is failed, you can get error code
        # and message from code and message.

        if response.status_code == HTTPStatus.OK:
            return response.output['text']  # 大模型输出结果: string
        else:
            return response.message     # api响应错误的返回信息: string
       
    def ernie_API(self, prompt, api_key = "rlZNNFVxD6xfF0oTyiz6iSG2", secret_key = "26NXkIxXoif4xR9X8GzFPA70NYTwWLDd"):
        url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions_pro?access_token=" + self.get_access_token(api_key, secret_key)
        
        payload = json.dumps({
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        })
        headers = {
            'Content-Type': 'application/json'
        }
        
        response = requests.request("POST", url, headers=headers, data=payload)

        if response.status_code == HTTPStatus.OK:
            return json.loads(response.text)['result']
        else:
            return response.error_msg   # 错误描述信息
        
    def spark_API(self, prompt, api_key = '33bc3a31df2d9c6f7a4ea13836588dfe', domain = "generalv3", Spark_url = "ws://spark-api.xf-yun.com/v3.1/chat", appid = 'e7141413', api_secret = 'Zjk5ZWNlYWVjNGQ0ZmFkNGMyMDAyZjZm'):
        response = spark_main(appid, api_key, api_secret, Spark_url, domain, prompt)
        return response
    
    def gpt3_5_API(self, prompt, key = "fk220564-ZWIbLd5laV0yWrdZxqoFJxbQO9JWwWDr"):
        url = "https://oa.api2d.net/v1/chat/completions"

        payload = json.dumps({
        "model": "gpt-3.5-turbo",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "safe_mode": False
        })
        headers = {
        'Authorization': f'Bearer {key}',
        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
        'Content-Type': 'application/json'
        }

        response = requests.request("POST", url, headers=headers, data=payload)
        response_text = eval(response.text.replace('null','None'))['choices'][0]['message']['content'].strip()
        return response_text
    
    def gpt_4_API(self, prompt, key = "fk220564-ZWIbLd5laV0yWrdZxqoFJxbQO9JWwWDr"):
        url = "https://oa.api2d.net/v1/chat/completions"

        payload = json.dumps({
        "model": "gpt-4-1106-preview",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "safe_mode": False
        })
        headers = {
        'Authorization': f'Bearer {key}',
        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
        'Content-Type': 'application/json'
        }

        response = requests.request("POST", url, headers=headers, data=payload)
        response_text = eval(response.text.replace('null','None'))['choices'][0]['message']['content'].strip()
        return response_text

def metrics_compute(preds, labels, all_labels):
    print('\n==========================================开始计算评价指标......==============================================')
    correct_match_nums, follow_insrtuction_nums = 0, 0
    for i in range(len(preds)):
        pred_tokens = preds[i]
        label = labels[i]
        match_correct, insrtuction_follow = text_classification_metrics(
            pred_tokens, label, all_labels)
        correct_match_nums += match_correct
        follow_insrtuction_nums += insrtuction_follow

    acc_match = correct_match_nums / len(labels)
    insrtuction_follow_precent = follow_insrtuction_nums / len(labels)

    return acc_match, insrtuction_follow_precent

def get_prompt(text,all_labels,examples,type):
    prompt_text_classification_zero_shot = f"""这是一个文本分类任务，现给定所有类别如下：
    {all_labels}

    请从所有类别中选择合适的一个作为下列输入文本的分类结果，无需其他解释和说明。
    输入文本：{text}

    分类结果："""
    prompt_text_classification_few_shot = f"""这是一个文本分类任务，现给定所有类别如下：
    {all_labels}

    参考示例如下：
    {examples}
    请参考上述示例从所有类别中选择合适的一个作为以下【文本】的分类结果，无需其他解释和说明。
    输入文本：{text}

    分类结果："""
    
    # prompt_text_classification_zero_shot = f"""给定一条测试数据："text": "{row['text']}",
    #                 请你给这个数据标注类别，可选labels有:
    #                 {all_labels}
                    
    #                 请回答哪个label是该条测试数据的标签，回复格式如下：
    #                 "该数据所属的label为：x"
    #                 """
    # prompt_text_classification_few_shot = f"""给定一条测试数据："text": "{row['text']}",
    #                 请你给这个数据标注类别，可选labels有:
    #                 {all_labels}
    #                 参考示例如下：
    #                 {examples}
                    
    #                 请回答哪个label是该条测试数据的标签，回复格式如下：
    #                 "该数据所属的label为：x"
    #                 """

    
    
    prompt_configs = {
        "文本分类":{
            "few_shot": prompt_text_classification_few_shot,
            "zero_shot": prompt_text_classification_zero_shot
        }
    }
    
    if type=="zero_shot":
        return prompt_configs["文本分类"][type]
    elif type=="few_shot":
        return prompt_configs["文本分类"][type]
    
if __name__=="__main__":
    MODEL_NAMEs = ['gpt3_5','gpt_4']
    Data2alllabels = {
        }
    Types = ["zero_shot","few_shot"]
    for model_idx,model_name in enumerate(MODEL_NAMEs):
        print("========================================================================================")
        print(f'MODEL:{model_name}')
        for data_path,all_labels in Data2alllabels.items():
            for type in Types:
                mkdir(f'{data_path}闭源测评/{type}/{model_name}')
               
                print("--------------------------------------------------------------------")
                test_data = pd.read_json(data_path+'test.json',lines=True)
                examples = pd.read_json(data_path+"dev.json",lines=True)[:5]
                if isinstance(all_labels,dict):
                    labels_dict = all_labels
                    all_labels = labels_dict.values()
                    labels = [labels_dict[label] for label in test_data['label']]
                elif isinstance(all_labels,list):
                    labels = test_data['label']
                responses = []
                for idx,row in tqdm(test_data.iterrows(), total=test_data.shape[0]):                  
                    prompt = get_prompt(row['text'],all_labels,examples,type)
                    instance = Model2FuncClass()
                    start = time.time()
                    response = getattr(instance, model_name+'_API')(prompt)
                    timeperline = time.time()-start
                    responses.append(response)
                    
                    line = {'prompt': prompt, 'llm_output': response, 'label': labels[idx], 'cost_time': timeperline}
                    with open(f'{data_path}/闭源测评/{type}/{model_name}/records.json', "a+", encoding="utf-8") as f:
                        json.dump(line, f, indent=4, ensure_ascii=False)
                    
                acc_match, insrtuction_follow_precent = metrics_compute(responses, labels, all_labels)
                print(f"model_name is {model_name}")
                print(f"type is {type}")
                print(f"data is {data_path}")
                print(f"acc_match is {acc_match}")
                print(f"insrtuction_follow_precent is {insrtuction_follow_precent}")
                print("--------------------------------------------------------------------")
                   
                with open(f'{data_path}/闭源测评/{type}/{model_name}/result.txt','a+',encoding='utf-8') as f:
                    f.write(f"model_name is {model_name}\n")
                    f.write(f"type is {type}\n")
                    f.write(f"data is {data_path}\n")
                    f.write(f"cost time per line id {timeperline}\n")
                    f.write(f"acc_match is {acc_match}\n")
                    f.write(f"insrtuction_follow_precent is {insrtuction_follow_precent}\n")
                    f.write('\n')
        print("========================================================================================")