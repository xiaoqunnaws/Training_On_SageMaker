# 大模型 SageMaker 训练框架

在 Amazon SageMaker 上进行大语言模型训练的完整解决方案，支持多种主流训练框架。

## 项目结构

```
├── llama_factory_on_sagemaker/    # LLaMA-Factory 训练框架
└── ms_swift_on_sagemaker/         # MS-Swift 训练框架
```

## 支持的训练框架

### LLaMA-Factory
- 基于 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 框架
- 支持 Pretrain/SFT/RLHF 训练
- 支持多机多卡和单机多卡训练

### MS-Swift
- 基于 [MS-Swift](https://github.com/modelscope/swift) 框架
- 支持多种模型的高效微调
- 集成 DeepSpeed 加速训练

## 快速开始

1. 选择训练框架目录
2. 参考对应的 README 文档
3. 修改训练配置参数
4. 提交 SageMaker Training Job

## 环境要求

- Amazon SageMaker
- Python 3.10+

## 许可证

本项目遵循各训练框架的原始许可证。