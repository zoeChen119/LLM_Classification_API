import os
import sys
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
    
    def qwen_API(self, prompt, api_key=''):
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
       
    def ernie_API(self, prompt, api_key = "", secret_key = ""):
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

        # if response.status_code == HTTPStatus.OK:
        try:
            return eval(response.text.replace('false','"false"'))['result']
        except:
            # return response.error_msg   # 错误描述信息
            return '无结果'
        
    def spark_API(self, prompt, api_key = '', domain = "generalv3", Spark_url = "ws://spark-api.xf-yun.com/v3.1/chat", appid = '', api_secret = ''):
        response = spark_main(appid, api_key, api_secret, Spark_url, domain, prompt)
        if '11200' in response:
            sys.exit()
        return response
        
    def gpt3_5_API(self, prompt, key = ""):
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
    
    def gpt_4_API(self, prompt, key = ""):
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
    print('\n--------------------开始计算评价指标......---------------------------')
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

def get_examples(dev_data,labels_dict=None,n_shot=5):
    # 从每个类别中采样一条数据作为示例
    example_df = dev_data.groupby('label').apply(lambda x: x.sample(n=1, random_state=24))
    # 剩下的数据
    rest_data = pd.concat([example_df, dev_data]).drop_duplicates(keep=False).reset_index(drop=True)

    if len(example_df) < n_shot:
        example_df_rest = rest_data.sample(n=(n_shot-len(example_df)), random_state=24)
        example_df = pd.concat([example_df, example_df_rest], axis=0).reset_index(drop=True)
    elif len(example_df) > n_shot:
        example_df = example_df.sample(n=n_shot, random_state=24).reset_index(drop=True)
    else: 
        example_df = example_df

    if example_df['label'].dtype=='int64':
        assert isinstance(labels_dict,dict)
        example_df['label'] = [labels_dict[label] for label in example_df['label']]
    example_list = []
    for i in range(len(example_df)):
        example = '输入文本：' + example_df['text'].iloc[i] + '\n' + '分类结果：'+ example_df['label'].iloc[i] + '\n\n'
        example_list.append(example)
    
    return example_list

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
    MODEL_NAMEs = ['ernie','gpt3_5','gpt_4', 'qwen','spark']
    Data2alllabels = {
        '/xy/ZOE/FINAL_Data/公开数据集/MUSIED-意图识别/':
            ['食品不熟', '使用回收食品', '感官异常', '异物', '包装破损', '宣传与实物不符', '餐凉', '身体不适', '变质/发霉', '质量问题', '临期/过期', '投诉商家经营环境'],
        '/xy/ZOE/FINAL_Data/公开数据集/CSDS-意图识别/':
            ['售后运费', '家电安装', '物流全程跟踪', '商品价格咨询', '少商品与少配件', '优惠券退回', '商家入驻条件', '礼品卡使用', '返修退换货处理周期', '售前运费多少', '手机邮件相关问题', '价保记录查询', '属性咨询', '能否自提', '白条使用流程', '什么时间出库', '补货时间', '优惠券获得方式', '查看发票', '订单状态解释', '发票退换修改', '物流损', '联系客户', '退款到哪儿', '恢复订单', '正常退款周期', '充值未到账充值到账时间', '货到付款', '拒收', '申请退款', '物流信息不正确', '近期活动咨询', '关闭服务单', '填写发票信息', '如何取消订单', '能否配送', '库存状态', '保修返修及退换货政策', '审核时效', '使用咨询', '订单签收异常', '价保申请流程', '是否提供发票', '预约配送时间', '补发票', '服务单修改', '增票相关', '订单无故取消', '返回方式', '价保条件', 'PLUS会员', '查询取消是否成功', '配送周期', '余额提现', '手机回收流程', '服务单查询', '配送方式', '联系配送', '修改订单'],
        '/Path/to/数据集1/':
            {1:'label1',0:'label2'},
        '/Path/to/数据集2/':
            {1:'label1',0:'label2'},       
    }
    Types = ["zero_shot","few_shot"]# "zero_shot","few_shot"
    
    for model_idx,model_name in enumerate(MODEL_NAMEs):
        print("========================================================================================")
        print(f'MODEL:{model_name}')
        for data_path,all_labels in Data2alllabels.items():
            print("--------------------------------------------------------------------")
            if data_path in ['/Path/to/数据集1/','/Path/to/数据集2/']:
                test_data = pd.read_json(data_path+'test.json',lines=False)
                dev_data = pd.read_json(data_path+"dev.json",lines=False)
            else:
                test_data = pd.read_json(data_path+'test.json',lines=True)
                dev_data = pd.read_json(data_path+"dev.json",lines=True)
            if isinstance(all_labels,dict):
                        labels_dict = all_labels
                        all_labels = labels_dict.values()
                        labels = [labels_dict[label] for label in test_data['label']]
                        examples = get_examples(dev_data,labels_dict)
            elif isinstance(all_labels,list):
                labels = test_data['label']
                examples = get_examples(dev_data)
                
            for type in Types:
                try:
                    RUN_STATUS = False
                    mkdir(f'{data_path}闭源测评/{type}/{model_name}')
                    
                    record_path = f'{data_path}闭源测评/{type}/{model_name}/records.json'
                    if os.path.exists(record_path):
                        print("File exists,尝试断点续运行...")
                        with open(f'{data_path}闭源测评/{type}/{model_name}/records.json','r', encoding='utf-8') as f:
                            lines = f.readlines()
                        print(f"已测评{len(lines)/6}行....")
                        if len(lines)/6<len(test_data):
                            print("尚未运行完,从该行开始续运行....")
                            test_data = test_data[int(len(lines)/6):]
                        else:
                            RUN_STATUS=True
                        
                    if RUN_STATUS==False:
                        for idx,row in tqdm(test_data.iterrows(), total=test_data.shape[0]):                  
                            prompt = get_prompt(row['text'],all_labels,examples,type)
                            instance = Model2FuncClass()
                            start = time.time()
                            response = getattr(instance, model_name+'_API')(prompt)
                            timeperline = time.time()-start
                            
                            line = {'prompt': prompt, 
                                    'llm_output': response, 
                                    'label': labels[idx], 
                                    'cost_time': timeperline}
                            with open(f'{data_path}闭源测评/{type}/{model_name}/records.json', "a+", encoding="utf-8") as f:
                                json.dump(line, f, indent=4, ensure_ascii=False)
                                f.write('\n')
                        RUN_STATUS=True
                    
                    if RUN_STATUS==True:
                        # if isinstance(all_labels,dict):
                        #     labels_dict = all_labels
                        #     all_labels = labels_dict.values()
                        
                        responses = []
                        labels = []
                        cost_times = []
                        with open(f'{data_path}闭源测评/{type}/{model_name}/records.json','r', encoding='utf-8') as f:
                            lines = f.readlines()
                            for i in range(0, len(lines), 6):
                                result = eval(''.join(lines[i:i+6]))
                                responses.append(result['llm_output'])
                                labels.append(result['label'])
                                cost_times.append(result['cost_time'])
                        
                    
                    avg_cost_time = sum(cost_times)/len(cost_times)
                    acc_match, insrtuction_follow_precent = metrics_compute(responses, labels, all_labels)
                    print(f"model_name is {model_name}")
                    print(f"type is {type}")
                    print(f"data is {data_path}")
                    print(f"acc_match is {acc_match}")
                    print(f"insrtuction_follow_precent is {insrtuction_follow_precent}")
                    print("--------------------------------------------------------------------")
                    
                        
                    with open(f'{data_path}闭源测评/{type}/{model_name}/result.txt','a+',encoding='utf-8') as f:
                        f.write(f"model_name is {model_name}\n")
                        f.write(f"type is {type}\n")
                        f.write(f"data is {data_path}\n")
                        f.write(f"cost time per line id {avg_cost_time}\n")
                        f.write(f"acc_match is {acc_match}\n")
                        f.write(f"insrtuction_follow_precent is {insrtuction_follow_precent}\n")
                        f.write('\n')
                except:
                    break
        print("========================================================================================")