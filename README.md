

# 实体关系抽取（RE）与知识图谱构建项目说明文档

## 1. 数据集描述

### 1.1 数据集概览
本项目使用结构化关系抽取数据集，包含：
- **训练集**：`train.jsonl`（1,366条）
- **测试集**：`test.jsonl`（370条）

### 1.2 数据格式
每条数据包含：
- `sentence`：包含实体关系的原始文本
- `relations`：关系三元组列表，每个三元组包含：
  - `head`：头实体（名称、类型、字符位置）
  - `type`：关系类型
  - `tail`：尾实体（名称、类型、字符位置）

**数据示例**：
```json
{
  "sentence": "A bio-inspired model [...] in VLSI.",
  "relations": [
    {
      "head": {"name": "bio-inspired model", "type": "NA", "pos": [2, 20]},
      "type": "used-for",
      "tail": {"name": "analog programmable array processor [...]", "pos": [28, 80]}
    },
    // 其他关系三元组
  ]
}
```
注：
1. "-LRB-/-RRB-"表示原始文本中的括号
2. 实体类型字段"NA"表示类型无关的设置

---

## 2. 模型选型与依据

### 2.1 模型列表
| 完整名称 | 参数量 | 架构类型 |
|----------|--------|----------|
| Qwen/Qwen2.5-0.5B-Instruct | 5亿 | 指令调优Transformer |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | 15亿 | 蒸馏版Qwen变体 |
| FacebookAI/roberta-base | 1.25亿 | BERT式编码器 |
| answerdotai/ModernBERT-base | 1.5亿 | 优化BERT变体 |

### 2.2 选型依据
- **Qwen2.5 & DeepSeek**：
  - 突出的指令跟随能力
  - 在生成式关系抽取任务中的SOTA表现
  - DeepSeek通过知识蒸馏实现高计算效率
  
- **RoBERTa**：
  - 作为强基线模型
  - 在分类式关系抽取中的成熟方案
  
- **ModernBERT**：
  - 采用层动态缩放（layer-wise scaling）
  - 改进的动态掩码策略
  - 在语义理解任务中的优化设计

---

## 3. 环境依赖

### 3.1 核心依赖包
```python
Python == 3.12.2
torch == 2.5.1
numpy == 1.26.4
transformers == 4.49.0
datasets == 3.3.1
sklearn-crfsuite == 0.5.0  # 指标计算
```

### 3.2 环境配置
#### 使用conda创建虚拟环境
```python
conda create -n re python=3.12 -y
conda activate re
```

## 4. 项目结构与运行
### 4.1 项目文件结构
```text
RE/
├── config.py        # 参数解析模块
├── Qwen_SFT.py      # 主训练脚本
├── Dpsk_SFT.py
├── roberta&ModernBERT.py 
├── train.jsonl      # 数据集
└── test.jsonl
```

### 4.2 Shell运行命令集
```bash
python Qwen_SFT.py \
  --model_name_or_path "Qwen/Qwen2.5-0.5B-Instruct" \
  --train_data "./data/train.jsonl" \
  --output_dir "./output" \
  --batch_size 4 \
  --grad_accum_steps 2 \
  --lr 3e-5 \
  --warmup_ratio 0.1 \
  --其他需要添加的参数
```

## 5. 实验设计

### 5.1 处理流程
```mermaid
graph TD
    A[数据加载] --> B[实体位置编码]
    B --> C[模型格式转换]
    C --> D[模型训练]
    D --> E[预测解码]
    E --> F[正则化处理]
    F --> G[指标计算]
```

### 5.2 关键技术
1. **数据预处理**：
   - 基于字符位置的实体跨度提取
   - 模型特定格式转换：
     * 生成式模型：提示工程（如"识别文本中的关系：[文本]"）
     * 编码器模型：BIO序列标注

2. **训练配置**：
   - 批量大小：生成式模型16，编码器模型32
   - 学习率：1e-4（线性预热）
   - 早停策略（容忍3个epoch）

3. **评估方法**：
   - 严格三元组匹配标准
   - 核心指标：
     * 准确率（Precision）：TP/(TP+FP)
     * 召回率（Recall）：TP/(TP+FN) 
     * F1值：2*(P*R)/(P+R)

4. **对抗实验**：
   - Qwen与DeepSeek交叉验证
   - 通过 `NEFTune` 方法向模型嵌入层注入噪声，参数阈值为3.0-5.0。

---

## 6. 实验结果与结论

### 6.1 性能对比
| 模型 | 噪声微调（NEFT） | 准确率(%) | 召回率(%) | F1值(%) |
|------|---------|-----------|-----------|---------|
| Qwen/Qwen2.5-0.5B-Instruct | without neft | 96.83 | 29.19 | 44.85 |
| Qwen/Qwen2.5-0.5B-Instruct | with neft | 87.65 | 33.97 | 48.97 |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | without neft | 98.61 | 33.97 | 50.53 |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | with neft | 97.22 | 33.49 | 49.82 |
| FacebookAI/roberta-base |  | 86.65 | 79.39 | 79.15 |
| answerdotai/ModernBERT-base |  | 86.96 | 79.55 | 80.17 |

### 6.2 核心指标计算公式

![Precision](https://i.upmath.me/svg/Precision%20%3D%20%5Cfrac%7B%5Ctext%7BTrue%20Positives%7D%7D%7B%5Ctext%7BTrue%20Positives%7D%20%2B%20%5Ctext%7BFalse%20Positives%7D%7D)

![Recall](https://i.upmath.me/svg/Recall%20%3D%20%5Cfrac%7B%5Ctext%7BTrue%20Positives%7D%7D%7B%5Ctext%7BTrue%20Positives%7D%20%2B%20%5Ctext%7BFalse%20Negatives%7D%7D)

![F1](https://i.upmath.me/svg/F1%5Ctext%7B-Score%7D%20%3D%202%20%5Ctimes%20%5Cfrac%7B%5Ctext%7BPrecision%7D%20%5Ctimes%20%5Ctext%7BRecall%7D%7D%7B%5Ctext%7BPrecision%7D%20%2B%20%5Ctext%7BRecall%7D%7D)

### 6.3 结论

1. **结果分析**  

   * 实验数据验证了Encoder-only模型（ModernBERT F1=80.17%，RoBERTa F1=79.15%）相比Decoder-only模型（DeepSeek-R1 F1=50.53%，Qwen2.5 F1=48.97%）在综合性能上的显著优势，该现象主要源于两类架构的本质差异。

2. **信息感知维度**  

   * Encoder的双向注意力机制展现出全局语义理解优势。在本项目中，ModernBERT的召回率（79.55%）是Qwen2.5（29.19%）的2.72倍，证明双向语境整合对关系推理的关键作用。尽管Decoder模型Qwen2.5达到96.83%的准确率，但其低召回率暴露了单向注意力的语义覆盖缺陷。 

3. **误差传播机制**  

   * Decoder架构的逐token生成特性导致错误累积：当实体识别出现偏差时，即实体对象被错误分割，Qwen2.5的关系召回率仅为29.19%，而ModernBERT通过独立分类设计实现79.55%的召回率，显示架构差异对错误传播的抑制作用。

4. **评价指标与生成式模型的契合**  

   * 生成式模型的开放输出特性与严格匹配的评价指标存在结构性矛盾。以Deepseek-r1为例，其生成结果虽在语义层面正确，但因文本表述灵活多样，在要求精确字符匹配的F1评分中表现不佳。这一现象凸显了传统评价体系难以全面衡量生成模型在关系抽取任务中的实际性能，需要引入考虑语义等价的评估方法。

5. **对抗试验分析**  

   * 生成式Decoder架构对训练噪声和实体识别误差具有双重敏感性：当添加NEFT较大时，Precision显著下降（98.61%→97.22%，Δ-1.39），而Recall微降（33.97%→33.49%，Δ-0.48），体现高精度模型对高噪声的脆弱性；但随着neft参数的逐渐下调，在实体识别任务中，生成式模型不论是召回率、精确率抑或是F1得分均稳步上升，并在到达某一值时出现了正向突破。该现象表明，适当的neft参数能够提升模型的评价指标。

---
