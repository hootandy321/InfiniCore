# InfiniCore

[![Doc](https://img.shields.io/badge/Document-ready-blue)](https://github.com/InfiniTensor/InfiniCore-Documentation)
[![CI](https://github.com/InfiniTensor/InfiniCore/actions/workflows/build.yml/badge.svg?branch=main)](https://github.com/InfiniTensor/InfiniCore/actions)
[![license](https://img.shields.io/github/license/InfiniTensor/InfiniCore)](https://mit-license.org/)
![GitHub repo size](https://img.shields.io/github/repo-size/InfiniTensor/InfiniCore)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/InfiniTensor/InfiniCore)

[![GitHub Issues](https://img.shields.io/github/issues/InfiniTensor/InfiniCore)](https://github.com/InfiniTensor/InfiniCore/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr/InfiniTensor/InfiniCore)](https://github.com/InfiniTensor/InfiniCore/pulls)
![GitHub contributors](https://img.shields.io/github/contributors/InfiniTensor/InfiniCore)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/InfiniTensor/InfiniCore)

InfiniCore 是一个跨平台统一编程工具集，为不同芯片平台的功能（包括计算、运行时、通信等）提供统一 C 语言接口。目前支持的硬件和后端包括：

- CPU；
- CUDA
  - 英伟达 GPU；
  - 摩尔线程 GPU；
  - 天数智芯 GPU；
  - 沐曦 GPU；
  - 曙光 DCU；
- 华为昇腾 NPU；
- 寒武纪 MLU；
- 昆仑芯 XPU；

API 定义以及使用方式详见 [`InfiniCore文档`](https://github.com/InfiniTensor/InfiniCore-Documentation)。

## 📚 完整开发文档 / Complete Documentation

**[English Documentation](docs/README.md)** - Comprehensive developer documentation

### 文档目录 / Documentation Index
- **[项目架构 / Architecture](docs/architecture.md)** - 系统设计与模块概述 / System design and module overview
- **[开发环境搭建 / Setup Guide](docs/setup.md)** - 构建与安装指南 / Build and installation guide  
- **[算子开发 / Operator Development](docs/operators.md)** - 自定义算子开发教程 / Custom operator development tutorial
- **[大模型适配 / Model Adaptation](docs/models.md)** - 大语言模型集成指南 (如Qwen3) / Large model integration guide (e.g., Qwen3)
- **[性能优化 / Performance](docs/performance.md)** - 跨平台性能优化技术 / Cross-platform performance optimization
- **[API参考 / API Reference](docs/api/README.md)** - 完整API文档 / Complete API documentation
- **[示例代码 / Examples](docs/examples/README.md)** - 实用代码示例 / Practical code examples
- **[故障排除 / Troubleshooting](docs/troubleshooting.md)** - 常见问题解决方案 / Common issues and solutions

### 快速开始 / Quick Start
1. **环境搭建** / **Setup**: 按照 [setup guide](docs/setup.md) 配置开发环境
2. **第一个算子** / **First Operator**: 学习 [operator development](docs/operators.md) 创建自定义算子  
3. **模型集成** / **Model Integration**: 使用 [model guide](docs/models.md) 适配大语言模型
4. **性能调优** / **Performance Tuning**: 参考 [performance guide](docs/performance.md) 优化性能

## 配置和使用

### 一键安装

在 `script/` 目录中提供了 `install.py` 安装脚本。使用方式如下：

```shell
cd InfiniCore

python scripts/install.py [XMAKE_CONFIG_FLAGS]
```

参数 `XMAKE_CONFIG_FLAGS` 是 xmake 构建配置，可配置下列可选项：

| 选项                     | 功能                              | 默认值
|--------------------------|-----------------------------------|:-:
| `--omp=[y\|n]`           | 是否使用 OpenMP                   | y
| `--cpu=[y\|n]`           | 是否编译 CPU 接口实现             | y
| `--nv-gpu=[y\|n]`        | 是否编译英伟达 GPU 接口实现       | n
| `--ascend-npu=[y\|n]`    | 是否编译昇腾 NPU 接口实现         | n
| `--cambricon-mlu=[y\|n]` | 是否编译寒武纪 MLU 接口实现       | n
| `--metax-gpu=[y\|n]`     | 是否编译沐曦 GPU 接口实现         | n
| `--moore-gpu=[y\|n]`     | 是否编译摩尔线程 GPU 接口实现     | n
| `--iluvatar-gpu=[y\|n]`  | 是否编译沐曦 GPU 接口实现         | n
| `--sugon-dcu=[y\|n]`     | 是否编译曙光 DCU 接口实现         | n
| `--kunlun-xpu=[y\|n]`    | 是否编译昆仑 XPU 接口实现         | n
| `--ninetoothed=[y\|n]`   | 是否编译九齿实现                 | n
| `--ccl=[y\|n]`           | 是否编译 InfiniCCL 通信库接口实现 | n

### 手动安装

0. 生成九齿算子（可选）

    参见[使用九齿](#使用九齿)章节。

1. 项目配置

   windows系统上，建议使用`xmake v2.8.9`编译项目。
   - 查看当前配置

     ```shell
     xmake f -v
     ```

   - 配置 CPU（默认配置）

     ```shell
     xmake f -cv
     ```

   - 配置加速卡

     ```shell
     # 英伟达
     # 可以指定 CUDA 路径， 一般环境变量为 `CUDA_HOME` 或者 `CUDA_ROOT`
     # window系统：--cuda="%CUDA_HOME%"
     # linux系统：--cuda=$CUDA_HOME
     xmake f --nv-gpu=true --cuda=$CUDA_HOME -cv

     # 寒武纪
     xmake f --cambricon-mlu=true -cv

     # 华为昇腾
     xmake f --ascend-npu=true -cv
     ```

2. 编译安装

   默认安装路径为 `$HOME/.infini`。

   ```shell
   xmake build && xmake install
   ```

3. 设置环境变量

   按输出提示设置 `INFINI_ROOT` 和 `LD_LIBRARY_PATH` 环境变量。

### 运行测试

#### 运行Python算子测试

```shell
python test/infiniop/[operator].py [--cpu | --nvidia | --cambricon | --ascend]
```

#### 一键运行所有Python算子测试

```shell
python scripts/python_test.py [--cpu | --nvidia | --cambricon | --ascend]
```

#### 算子测试框架

详见 `test/infiniop-test` 目录

#### 通信库（InfiniCCL）测试

编译（需要先安装InfiniCCL）：

```shell
xmake build infiniccl-test
```

在英伟达平台运行测试（会自动使用所有可见的卡）：

```shell
infiniccl-test --nvidia
```

### 使用九齿

[九齿](https://github.com/InfiniTensor/ninetoothed)是一门基于 Triton 但提供更高层抽象的领域特定语言（DSL）。使用九齿可以降低算子的开发门槛，并且提高开发效率。

InfiniCore 目前已经可以接入使用九齿实现的算子，但是这部分实现的编译是默认关闭的。如果选择编译库中的九齿实现，需要使用 `--ninetoothed=y`，并在运行一键安装脚本前完成以下准备工作：

1. 安装九齿与[九齿算子库](https://github.com/InfiniTensor/ntops)：

```shell
git clone https://github.com/InfiniTensor/ntops.git
cd ntops
pip install -e .
```

注：安装 `ntops` 时，`ninetoothed` 会被当成依赖也一并安装进来。

2. 在 `InfiniCore` 文件夹下运行以下命令 AOT 编译库中的九齿算子：

```shell
PYTHONPATH=${PYTHONPATH}:src python scripts/build_ntops.py
```

注：如果对九齿相关文件有修改，需要重新构建 InfiniCore 时，也需要同时运行以上命令进行重新生成。

3. 按照上面的指引进行[一键安装](#一键安装)或者[手动安装](#手动安装)。

## 如何开源贡献

见 [`InfiniCore开发者手册`](DEV.md)。
