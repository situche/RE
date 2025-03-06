import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="实体关系抽取模型训练参数")
    
    # 数据参数
    parser.add_argument("--train_data", type=str, required=True,
                       help="训练数据文件路径（JSONL格式）")
    parser.add_argument("--eval_data", type=str, required=True,
                       help="验证数据文件路径（JSONL格式）")
    
    # 模型参数
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                      help="预训练模型名称或路径")
    
    # 训练参数组
    train_group = parser.add_argument_group("训练配置")
    train_group.add_argument("--output_dir", type=str, default="./output",
                            help="模型和日志输出目录")
    train_group.add_argument("--learning_rate", type=float, default=5e-5,
                            help="学习率（建议1e-5 ~ 5e-5）")
    train_group.add_argument("--per_device_train_batch_size", type=int, default=4,
                            choices=[2, 4, 8, 16], help="训练批次大小（根据显存调整）")
    train_group.add_argument("--per_device_eval_batch_size", type=int, default=8,
                            help="验证批次大小")
    train_group.add_argument("--num_train_epochs", type=int, default=15,
                            help="训练总轮次")
    train_group.add_argument("--gradient_accumulation_steps", type=int, default=2,
                            help="梯度累积步数（显存不足时使用）")
    
    # 优化器参数
    optim_group = parser.add_argument_group("优化配置")
    optim_group.add_argument("--weight_decay", type=float, default=0.01,
                            help="权重衰减系数")
    optim_group.add_argument("--warmup_ratio", type=float, default=0.1,
                            help="学习率预热比例")
    
    # 评估参数
    eval_group = parser.add_argument_group("评估配置")
    eval_group.add_argument("--evaluation_strategy", type=str, default="epoch",
                           choices=["steps", "epoch"], help="评估策略")
    eval_group.add_argument("--eval_steps", type=int, default=500,
                           help="当策略为steps时的评估间隔")
    eval_group.add_argument("--metric_for_best_model", type=str, default="f1",
                           choices=["f1", "precision", "recall"],
                           help="早停依据指标")
    
    # 保存配置
    save_group = parser.add_argument_group("保存配置")
    save_group.add_argument("--save_strategy", type=str, default="epoch",
                           choices=["steps", "epoch", "no"],
                           help="模型保存策略")
    save_group.add_argument("--save_steps", type=int, default=500,
                           help="当策略为steps时的保存间隔")
    save_group.add_argument("--save_total_limit", type=int, default=1,
                           help="最大保存检查点数量")
    save_group.add_argument("--load_best_model_at_end", action="store_true",
                           help="训练结束时加载最佳模型")
    
    # 系统配置
    sys_group = parser.add_argument_group("系统配置")
    sys_group.add_argument("--fp16", action="store_true",
                          help="启用混合精度训练（NVIDIA GPU）")
    sys_group.add_argument("--bf16", action="store_true",
                          help="启用BF16精度（仅限Ampere+架构GPU）")
    sys_group.add_argument("--log_level", type=str, default="warning",
                          choices=["debug", "info", "warning", "error"],
                          help="日志级别")
    
    return parser.parse_args()
