import re

def text_classification_metrics(predict, label, all_labels):
    # try:
    #     match = re.search(r"\d", predict)
    #     predict_label = match.group()
    #     predict_label = int(predict_label.replace(' ', '').replace('\n', ''))  # 去除空格和换行
    # except:
    #     return 0,0   
     
    # try:
    #     predict_label = re.findall(r"该数据所属的label为：(.*)", predict)[0]
    # except:
    #     return 0,0 
    
    if predict not in all_labels:
        return 0,0
    else:
        predict_label = predict

    def _acc_match():
        if predict_label == label:    # 判断是否完全==正确标签
            match_correct = 1
        else:
            match_correct = 0
        return match_correct
        
    def _instruction_follow():
        # 计算指令跟随能力
        if predict_label in all_labels:  # 判断输出是否为所有标签中的一个
            insrtuction_follow  = 1
        else:
            insrtuction_follow = 0
        return insrtuction_follow
    
    match_correct = _acc_match()
    # if (all_label_embeds!=None) and (embed_model!=None):
    #     embed_correct = _acc_embed()
    insrtuction_follow = _instruction_follow()

    return match_correct, insrtuction_follow