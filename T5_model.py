import re
import json
import numpy as np
from datasets import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments, EarlyStoppingCallback, GenerationConfig
from sklearn.metrics import f1_score, recall_score, accuracy_score
from collections import OrderedDict

# 初始化T5模型和分词器
model_name = "google-t5/t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
model.config.decoder_start_token_id = tokenizer.pad_token_id

# 示例数据（这里的数据你已经提供，不需要全部重新写）
train_data = "/data/wangbin/No3/dataset/train.jsonl"
test_data = "/data/wangbin/No3/dataset/test.jsonl"

# 数据加载函数（确保relations转换为JSON格式）
def load_data(path):
    data = []
    with open(path,'r') as f:
        for i in f:
            line = json.loads(i)
            data.append(line)
    return data

def process_data(data):
    result = [] # 每个元素是该句子所有关系的 JSON 数组字符串
    for item in data:
        sentence = item["sentence"]
        en_rel = []
        for relation in item["relations"]:
            head = relation['head']['name'].strip()
            rel_type = relation['type'].strip()
            tail = relation['tail']['name'].strip()
            en_rel.append({
                "head": head,
                "type": rel_type,
                "tail": tail
            })
        # 转换为合法的 JSON 数组字符串（注意缩进为 None）
        relations_str = json.dumps(en_rel, ensure_ascii=False, indent=None)
        # print('relations_str: ', relations_str, '\t', 'type: ', type(relations_str))
        result.append({'sentence': sentence, 'relation': relations_str})
    # print(result[2])
    return result

train_dict = process_data(load_data(train_data))
test_dict = process_data(load_data(test_data))
train_dataset = Dataset.from_list(train_dict)
print(train_dataset)
test_dataset = Dataset.from_list(test_dict)

# 添加系统提示语句
system_prompt = '''You are a professional information extraction model. Please directly extract all entity relationship triplets from the input text and output only a JSON array in the exact format below. Do not include any additional text:

[
  {"head": "<head>", "type": "<type>", "tail": "<tail>"}
]

【Important】Strictly adhere to the following requirements:
1. Extract only explicitly expressed relationships from the text - do not infer speculative conclusions;
2. Use simple verbal phrases for relationship types (e.g.: "conjunction", "feature-of", "hyponym-of", "used-for", "part-of", "compare", "evaluate-for");
3. Maintain the exact key order shown in the above JSON format and ensure valid JSON syntax.

sentence: '''

# Tokenizer处理函数
def preprocess_function(examples):
    sentences = examples['sentence']
    targets = examples['relation']
    prompt_sentence = [system_prompt + sentence for sentence in sentences]
    model_inputs = tokenizer(prompt_sentence, max_length=256, truncation=True, padding="max_length",return_tensors='pt')
    labels = tokenizer(targets, max_length=256, truncation=True, padding="max_length", return_tensors='pt', add_special_tokens=True)["input_ids"]
    labels = labels.masked_fill(labels == tokenizer.pad_token_id, -100)
    model_inputs["labels"] = labels
    return model_inputs

# 处理数据集
tokenized_train = train_dataset.map(preprocess_function, batched=True, remove_columns=["sentence", "relation"])
tokenized_test = test_dataset.map(preprocess_function, batched=True, remove_columns=["sentence", "relation"])

def extract_triplets(text):
    """改进版正则表达式，支持更灵活的格式"""
    # 匹配键值对模式，允许三种情况：
    # 1. 带引号的键和值："head": "value"
    # 2. 无引号的键和值：head: value
    # 3. 混合情况
    pattern = r'''
    (?:[{[]\s*)?  # 允许开头有[{符号
    (?:["']?)(head|type|tail)(?:["']?)\s*[:=]\s*  # 键部分
    (?:["']?)([^"'{}[\],]+)(?:["']?)  # 值部分（排除特殊符号）
    (?=\s*[,}\]])  # 确保后面有结束符号
    '''

    matches = re.findall(pattern, text, re.VERBOSE | re.IGNORECASE)
    
    # 重组匹配结果
    triples = []
    current = {}
    for k, v in matches:
        if k.lower() in ['head', 'type', 'tail']:
            current[k.lower()] = v.strip()
            if len(current) == 3:
                if all(current.values()):  # 确保三个字段都有值
                    triples.append(current)
                current = {}

    # 过滤重复和无效数据
    seen = set()
    return [t for t in triples 
           if len(t) == 3 
           and (t['head'], t['type'], t['tail']) not in seen 
           and not seen.add((t['head'], t['type'], t['tail']))]

def compute_metrics(pred):
    preds, labels = pred.predictions, pred.label_ids

    # 解码处理保持不变
    preds = np.where((preds >= 0) & (preds < tokenizer.vocab_size), preds, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    print('decoded_preds: ', decoded_preds)
    # print('decoded_labels: ', decoded_labels)
    pred_relations = [extract_triplets(p) for p in decoded_preds]
    true_relations = [extract_triplets(l) for l in decoded_labels] 
    # print('pred_relations: ', pred_relations)
    print('true_relations: ', true_relations)
    # 三元组级别评估
    TP, FP, FN = 0, 0, 0
    for preds, truths in zip(pred_relations, true_relations):
        # 转换三元组为标准化集合
        pred_set = set()
        for rel in preds:
            head = rel.get('head', '').strip()
            rel_type = rel.get('type', '').strip()
            tail = rel.get('tail', '').strip()
            if head and rel_type and tail:
                pred_set.add((head, rel_type, tail))

        truth_set = set()
        for rel in truths:
            head = rel.get('head', '').strip()
            rel_type = rel.get('type', '').strip()
            tail = rel.get('tail', '').strip()
            if head and rel_type and tail:
                truth_set.add((head, rel_type, tail))

        TP += len(pred_set & truth_set)
        FP += len(pred_set - truth_set)
        FN += len(truth_set - pred_set)

    # 计算指标
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'f1-score': f1  # 保持与原代码兼容
    }

# 训练参数配置
training_args = Seq2SeqTrainingArguments(
    output_dir="/data/wangbin/No3/T5/output",  # 模型输出目录
    evaluation_strategy="steps",        # 按步数评估
    # eval_steps=500,                     # 每500步评估一次
    save_strategy="steps",              # 保存策略与评估对齐
    # save_steps=500,                     # 保存间隔与评估相同
    learning_rate=3e-5,                 # 适合生成任务的较小学习率
    per_device_train_batch_size=8,      # 根据显存调整批次大小
    per_device_eval_batch_size=16,      # 评估批次可适当放大
    gradient_accumulation_steps=2,      # 梯度累积应对大batch需求
    num_train_epochs=10,                # 最大训练轮次
    warmup_ratio=0.1,                   # 热身比例
    logging_dir="/data/wangbin/No3/T5/logs",               # 日志目录
    logging_steps=100,                  # 每100步记录日志
    load_best_model_at_end=True,        # 训练结束时加载最佳模型
    # metric_for_best_model="f1",         # 以F1作为早停依据
    predict_with_generate=True,         # 必须开启生成模式
    generation_max_length=256,          # 与预处理长度一致
)

# 初始化训练器
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# 训练模型
trainer.train()

# 保存模型、权重和分词器
model.save_pretrained("/data/wangbin/No3/T5/output")
tokenizer.save_pretrained("/data/wangbin/No3/T5/output")
