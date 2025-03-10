import re
import json
import torch
import numpy as np
from datasets import Dataset
from config import parse_args
from scipy.special import softmax
from sklearn.metrics import accuracy_score, f1_score, recall_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback

# 配置参数
# MODEL_NAME = "FacebookAI/roberta-base"
MODEL_NAME = "answerdotai/ModernBERT-base"
TRAIN_PATH = "data_path"
TEST_PATH = "data_path"
MAX_LENGTH = 256

# 初始化全局组件
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
relation_types = []
args = parse_args()

# 保证结果可重复
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
seed_everything(42)

# 数据加载与预处理
def load_data(path):
    """加载原始数据并构建关系类型字典"""
    global relation_types
    data = []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            # 提取所有关系类型
            for rel in item["relations"]:
                if rel["type"] not in relation_types:
                    relation_types.append(rel["type"])
            data.append(item)
    return data

def encode_entities(sentence, entities):
    """将实体位置编码为特殊标记"""
    # 按位置倒序插入避免干扰
    sorted_ents = sorted(entities, key=lambda x: -x["pos"][0])
    text = sentence
    for ent in sorted_ents:
        start, end = ent["pos"]
        text = text[:start] + f"<e:{ent['name']}>" + text[end:]
    return text

def process_to_dict(data):
    """处理为模型可用的字典格式"""
    processed = []
    for item in data:
        # 编码所有实体
        all_entities = [rel["head"] for rel in item["relations"]] + \
                      [rel["tail"] for rel in item["relations"]]
        encoded_text = encode_entities(item["sentence"], all_entities)
        
        # 生成每个关系的训练样本
        for rel in item["relations"]:
            # 添加特殊关系标记
            marked_text = f"<head>{rel['head']['name']}</head> " + \
                          f"<tail>{rel['tail']['name']}</tail> " + \
                          encoded_text
            
            processed.append({
                "text": marked_text,
                "label": relation_types.index(rel["type"])
            })
    return processed

# 数据编码
def tokenize_function(examples):
    """将文本转换为模型输入"""
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=256
    )

# 正则解析（适配分类任务）
def parse_predictions(logits):
    """将模型输出转换为可读格式"""
    pred_labels = np.argmax(logits, axis=1)
    return [{
        "relation": relation_types[label],
        "confidence": float(np.max(softmax(logit)))
    } for label, logit in zip(pred_labels, logits)]

# 评估指标计算
def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    return {
        "accuracy": accuracy_score(p.label_ids, preds),
        "f1_macro": f1_score(p.label_ids, preds, average="macro"),
        "recall_macro": recall_score(p.label_ids, preds, average="macro")
    }

# 训练流程
def main():
    # 数据加载与处理
    train_data = process_to_dict(load_data(TRAIN_PATH))
    test_data = process_to_dict(load_data(TEST_PATH))
    
    # 创建数据集
    train_dataset = Dataset.from_list(train_data).map(tokenize_function, batched=True)
    test_dataset = Dataset.from_list(test_data).map(tokenize_function, batched=True)
    
    # 动态设置分类器
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(relation_types),
    )
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy=args.evaluation_strategy,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        logging_dir=args.logging_dir,
        save_strategy=args.save_strategy,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
    )
    
    # 初始化Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    
    # 开始训练
    trainer.train()
    
    # 生成预测样例
    sample = test_dataset[0]
    with torch.no_grad():
        logits = model(torch.tensor([sample["input_ids"]]).to(model.device)).logits
    print("预测结果:", parse_predictions(logits.cpu().numpy()))
    
    # 保存模型
    model.save_pretrained("./output")
    tokenizer.save_pretrained("./output")

if __name__ == "__main__":
    main()
