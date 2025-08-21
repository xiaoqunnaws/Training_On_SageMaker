# MS-Swift on SageMaker

基于 [MS-Swift](https://github.com/modelscope/swift) 框架在 Amazon SageMaker 上进行大语言模型训练。

## 项目结构

```
├── entry.py                      # 训练任务入口文件
├── requirements.txt              # 依赖包列表
├── train_script_sagemaker.sh     # 训练启动脚本
├── training_on_sagemaker.ipynb   # SageMaker 训练示例
├── example_data.json             # 示例训练数据
├── s5cmd                         # S3 文件传输工具
└── ms-swift/                     # MS-Swift 框架代码
```

## 快速开始

### 0. 代码下载
```
git clone https://github.com/xqun3/Training_On_SageMaker.git
cd Training_On_SageMaker/

# ms-swift（v3.7.1） 以 submodule 的形式关联到当前 repo
git submodule init
git submodule update

# 进入 ms_swift_on_sagemaker 文件夹进行后续模型训练步骤
cd ms_swift_on_sagemaker/
```

按照 training_on_sagemaker.ipynb 里的步骤进行训练作业的提交

### 1. 准备训练数据
将训练数据上传到 S3，格式参考 `example_data.json`

### 2. 配置环境变量
在 SageMaker notebook 中设置：
- `MODEL_S3_PATH`: 基础模型 S3 路径
- `MODEL_LOCAL_PATH`: 本地模型路径
- 其他训练参数

### 3. 提交训练任务
最后提交 Training Job

## 主要特性

- **多机多卡训练**: 支持分布式训练
- **DeepSpeed 集成**: 高效内存管理和加速
- **自动模型同步**: S3 与本地自动同步
- **灵活配置**: 支持各种训练参数调整

## 依赖包

- ms-swift[all]==3.7.1
- torch==2.7.1
- deepspeed==0.15.4
- sagemaker-ssh-helper

## 训练流程

1. 从 S3 下载基础模型
2. 设置分布式训练环境
3. 执行 MS-Swift 训练脚本
4. 保存训练结果到 checkpoints

## 注意事项

- 确保 S3 权限配置正确
- 根据数据量调整实例类型
- 监控训练过程中的资源使用情况