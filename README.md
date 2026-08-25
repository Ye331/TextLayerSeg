# 重叠文字图层分割

本项目提供一个精简、可审计的上下文增强特征金字塔实现，用于四类重叠文字分割：

`RGB → ResNet-50 C2:C5 → C5 多膨胀率上下文 → 自顶向下 FPN → P2 解码器 → BG/T1/T2/OV`

| 编号 | 类别 | 含义 |
|---:|---|---|
| 0 | BG | 背景 |
| 1 | T1 | 仅属于第一文字层的可见像素 |
| 2 | T2 | 仅属于第二文字层的可见像素 |
| 3 | OV | 两个文字层发生重叠的像素 |

完整图层掩码可确定性地恢复为 `T1 ∪ OV` 和 `T2 ∪ OV`。本项目研究的是像素级图层分割，不主张能够提高 OCR 性能，也不主张可以恢复被完全遮挡的字符。

## 项目包含内容

- 最终选定的 ResNet-50 + FPN + Context 模型；
- 固定的四类分割损失与评测指标；
- JSONL 数据清单读取接口；
- 精简训练、评测、单图预测和合成数据 smoke test；
- 单元测试与经过脱敏的架构消融记录。

本仓库有意排除了数据集、字体、预训练权重、训练 checkpoint、S/W 几何分支、attention 特征传播、selector、阈值后处理和私有实验路径。

## 安装

项目要求 Python 3.10 或更高版本。请先根据自己的 CUDA/CPU 环境安装 PyTorch，再安装本项目：

```bash
pip install -e .
pip install pytest
```

## 快速验证

```bash
python scripts/validate_release.py
python scripts/smoke_test.py
pytest -q
```

完整发布模型包含 **34,737,868 个可训练参数**。为了缩短 CPU 检查时间，smoke test 使用较窄的 FPN；单元测试还会检查完整模型的参数量。

## 无版权依赖的玩具实验

玩具数据生成器会创建简单的交叉笔画及对应四类掩码，仅用于检查代码执行和数据接口，不属于正式 benchmark。

```bash
python examples/make_toy_data.py --output toy_data --size 128
python scripts/train.py \
  --train-manifest toy_data/train.jsonl \
  --val-manifest toy_data/val.jsonl \
  --output-dir runs/toy \
  --image-size 128 --batch-size 1 --workers 0 --epochs 1 --max-steps 2
python scripts/evaluate.py \
  --manifest toy_data/val.jsonl \
  --checkpoint runs/toy/best.pt \
  --image-size 128
```

## 数据格式

JSONL 文件的每一行包含一张 RGB 图像和一张单通道 PNG 掩码：

```json
{"id":"sample-001","image":"images/sample-001.png","mask":"masks/sample-001.png"}
```

默认情况下，相对路径以 manifest 所在目录为基准。掩码值必须为 `0..3`，数值 `255` 会被映射为损失函数的忽略标签。详细约定见 [`DATA_FORMAT.md`](DATA_FORMAT.md)。

## 训练与推理

发布实验使用的主要参数记录在 [`configs/fpn_context.json`](configs/fpn_context.json) 中。仓库内的训练器是用于检查模型和数据接口的精简单进程参考实现，并不等同于原始四卡、尺度鲁棒数据增强训练流程。参考命令如下：

```bash
python scripts/train.py \
  --train-manifest /path/to/train.jsonl \
  --val-manifest /path/to/dev.jsonl \
  --backbone-checkpoint /path/to/resnet50.pth \
  --output-dir runs/fpn_context
```

预测单张图像并输出标签 PNG：

```bash
python scripts/predict.py \
  --image input.png --checkpoint runs/fpn_context/best.pt --output prediction.png
```

骨干网络加载器兼容常见的 torchvision ResNet-50 state-dict 封装格式，并会忽略分类头参数。

## 已记录的实验结果

冻结的 seed-2026 分组开发集消融使用 120 个源页面训练，在另外 14 个无交集源页面上开发验证。完成 8,192 次有效更新后，C2-only、FPN 和 FPN+Context 的 mIoU 分别为 0.7202、0.7708 和 0.7800；最终选择的 FPN+Context 模型 OV IoU 为 0.6395。

以上结果仅属于**单随机种子内部开发集结果**，不是 official validation、official test、跨域结果或 SOTA 主张。由于数据和 checkpoint 的发布授权尚未完成，本仓库不重新分发这些文件。证据边界见 [`evidence/VALIDATION.md`](evidence/VALIDATION.md)。

## 目录结构

```text
pyramid_overlap/   模型、数据、损失、指标和 checkpoint 接口
scripts/           静态检查、smoke test、训练、评测和预测
examples/          manifest 示例与玩具数据生成器
configs/           固定的发布参数
tests/             模型、指标和数据接口测试
evidence/          预注册与脱敏实验结果
```

## 引用

本代码发布不声明已有正式发表论文。待相关论文公开后，可在此补充经过核验的引用信息。
