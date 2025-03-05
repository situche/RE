import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import json
from datasets import Dataset

# 模型加载
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# 加载分词器及填充方式
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = 'left'

system_prompt = '''你是一个专业的信息抽取模型。请从输入的文本中直接提取出所有的实体关系三元组，且仅输出符合下面格式的JSON数组，不要添加任何其他文字：

[
  {"head": "<head>", "type": "<type>", "tail": "<tail>"}
]

【注意】请严格遵循以下要求：
1. 只提取文本中明确表达的关系，不进行推理；
2. 关系类型使用简单动词短语（例如："conjunction", "feature-of", "hyponym-of", "used-for", "part-of", "compare", "evaluate-for"）；
3. 严格按照上面的格式输出，保证键的位置顺序与上面JSON格式中的键的顺序完全一致，确保JSON格式正确。
'''

# 数据加载
def load_data(path):
    data = []
    with open(path,'r') as f:
        for i in f:
            line = json.loads(i)
            data.append(line)
    return data

# 数据处理
def data_process(data):
    results = []
    for item in data:
        sentence = item['sentence']
        relations = []
        for rel in item["relations"]:
            head = rel['head']['name']
            rel_type = rel['type']
            tail = rel['tail']['name']
            relations.append({
                'head': head,
                'type': rel_type,
                'tail': tail
            })
        results.append({'sentence': sentence, 'relations': relations})
    return results

def data_encoding(data):
    encodings = {
        'input_ids': [],
        'attention_mask': [],
        'labels': []
    }
    for sentence, relations in zip(data['sentence'], data['relations']):
        user_prompt = sentence
        labels = json.dumps(relations)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": labels}
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = tokenizer(
            text,
            max_length=256,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        labels = tokenizer(
            labels,
            max_length=256,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        labels_input_ids = labels["input_ids"]
        labels_input_ids[labels_input_ids == tokenizer.pad_token_id] = -100
        
        encodings['input_ids'].append(model_inputs['input_ids'].squeeze(0))
        encodings['attention_mask'].append(model_inputs['attention_mask'].squeeze(0))
        encodings["labels"].append(labels['input_ids'].squeeze(0))

    return encodings

def parse_and_normalize(text, max_length=2048):
    """全能解析函数（安全、高效、无递归错误）"""
    # 预处理层（保障基础安全）
    text = str(text)[:max_length].strip()
    if not text or text.lower() in {'', 'n/a', '{}', '[]'}:
        return []

    # 统一清洗函数
    clean = lambda s: re.sub(r'\s+', ' ', str(s).strip(' .,;*#\n\t"\'\\')).lower()
    
    # 修复常见JSON错误
    text = re.sub(r'(?<!\\)\\(?!["\\/bfnrt])', '', text)  # 移除非法转义
    text = re.sub(r',\s*(?=[}\]])', '', text)  # 移除尾部逗号
    text = re.sub(r"'\s*:", '" :', text, flags=re.IGNORECASE)  # 统一键名引号

    # 阶段1：结构化解析（JSON/列表）
    results = []
    json_objs = re.finditer(
        r'\{(?:[^{}"\'\\]|\\["\'\\]|"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\')*?\}',
        text, 
        re.DOTALL
    )
    for match in json_objs:
        try:
            data = json.loads(match.group())
            if isinstance(data, dict):
                results.append({
                    "head": clean(data.get("head", "")),
                    "type": clean(data.get("type", "")),
                    "tail": clean(data.get("tail", ""))
                })
            elif isinstance(data, list):
                results.extend([{
                    "head": clean(t.get("head", "")),
                    "type": clean(t.get("type", "")),
                    "tail": clean(t.get("tail", ""))
                } for t in data if isinstance(t, dict)])
        except json.JSONDecodeError:
            continue

    # 阶段2：键值对扫描（保障非结构化数据）
    if not results:
        current = {}
        for match in re.finditer(
            r'\b(head|type|tail)\b\s*[:=]\s*'
            r'(?:"((?:[^"\\]|\\.)*)"|\'((?:[^\'\\]|\\.)*)\'|([^,}\]]+))',
            text,
            re.IGNORECASE
        ):
            key = match.group(1).lower()
            value = (match.group(2) or match.group(3) or match.group(4) or "").strip()
            current[key] = clean(value)
            if len(current) == 3:
                results.append(current)
                current = {}

    # 去重与验证层
    seen = set()
    return [
        r for r in results 
        if (k := (r["head"], r["type"], r["tail"])) 
        and k not in seen 
        and not seen.add(k)
    ]

# 定义评估函数
def compute_metrics(eval_preds):
    pred_ids, label_ids = eval_preds
    pred_ids = pred_ids.argmax(-1)

    label_ids[label_ids == -100] = tokenizer.pad_token_id

    decoded_preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    tp, fp, fn = 0, 0, 0

    for pred_str, label_str in zip(decoded_preds, decoded_labels):
        # 统一使用parse_and_normalize解析预测结果
        pred_triplets = parse_and_normalize(pred_str)
        
        # 改进后的真实标签解析逻辑
        try:
            # 优先使用统一解析函数
            true_triplets = parse_and_normalize(label_str)
            
            # 兜底逻辑：处理旧格式数据
            if not true_triplets:
                raw_label = json.loads(label_str)
                true_triplets = [{
                    "head": str(rel.get("head", "")).lower().strip(),
                    "type": str(rel.get("type", "")).lower().strip(),
                    "tail": str(rel.get("tail", "")).lower().strip()
                } for rel in raw_label]
                
        except Exception as e:
            print(f"Label解析错误: {str(e)}")
            true_triplets = []

        # 转换为可哈希集合
        pred_set = { (t["head"], t["type"], t["tail"]) for t in pred_triplets }
        true_set = { (t["head"], t["type"], t["tail"]) for t in true_triplets }

        tp += len(pred_set & true_set)
        fp += len(pred_set - true_set)
        fn += len(true_set - pred_set)

    # 计算指标（保持不变）
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-9)

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "support": len(decoded_labels)
    }


train_data = load_data("./data")
test_data = load_data("./data")
# print(train_data)
train_dict = data_process(train_data)
test_dict = data_process(test_data)
# print(train_dict)
train_dataset = Dataset.from_list(train_dict)
# print(train_dataset)
test_dataset = Dataset.from_list(test_dict)

tokenized_train = data_encoding(train_dataset)
tokenized_test = data_encoding(test_dataset)

train_dataset = Dataset.from_dict(tokenized_train)
test_dataset = Dataset.from_dict(tokenized_test)

# 定义训练参数
training_args = TrainingArguments(
    output_dir="./output",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=1,
    num_train_epochs=15,
    weight_decay=0.01,
    logging_dir='./logs',
    save_steps=500,
    save_total_limit=1,
)

# 定义Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,  # 将评估函数传递给Trainer
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

# 开始训练
trainer.train()

# 保存模型和分词器
model.save_pretrained("./output")
tokenizer.save_pretrained("./output")

print("Fine-tuned model and tokenizer have been saved.")
