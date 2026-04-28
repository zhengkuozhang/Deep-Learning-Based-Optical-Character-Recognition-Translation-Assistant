# 基于深度学习的OCR翻译助手 | Deep Learning-Based Optical Character Recognition Translation Assistant

基于深度学习的高精度课本文字提取与自动化翻译辅助工具。本项目采用赛博极简主义UI 设计，通过先进的计算机视觉模型，为语言学习和外文文献阅读提供沉浸式、自动化的翻译体验。

## 核心特性

- **深度学习驱动**：底层集成 EasyOCR，基于CRAFT文字检测与CRNN文字识别架构，精准识别复杂排版的课本截图。

- **零配置翻译**：内置 deep-translator 免费调用 Google Translate 接口，开箱即用，支持德语/英语到中文的高效翻译。

- **赛博极简美学**：基于 Streamlit 深度定制全局 CSS，呈现羊皮纸质感、优雅的衬线字体（Noto Serif SC / Crimson Pro）与现代化交互的完美融合。

- **易部署与跨平台**：纯 Python 技术栈，默认使用 CPU 推理，兼容 macOS/Windows/Linux，具备极强的便携性。

## 技术栈

- **前端与交互**：Streamlit

- **OCR 视觉引擎**：EasyOCR (PyTorch 后端)

- **图像处理矩阵**：Pillow, NumPy

- **自然语言翻译**：deep-translator

## 快速启动 (Quick Start)

### 1. 克隆项目

```bash
git clone https://github.com/zhengkuozhang/Deep-Learning-Based-Optical-Character-Recognition-Translation-Assistant.git
cd Deep-Learning-Based-Optical-Character-Recognition-Translation-Assistant
```

### 2. 安装依赖

建议在 Python 3.9 ~ 3.11 的虚拟环境（如 Conda 或 venv）中运行。

```bash
pip install -r requirements.txt
```

（注：默认依赖支持 CPU 推理。若需启用 GPU 加速，请前往 PyTorch 官网安装对应 CUDA 版本的环境，并在 app.py 中将 gpu=False 修改为 gpu=True）

### 3. 启动应用

```bash
streamlit run app.py
```

运行成功后，浏览器将自动弹窗并打开 Web 界面,默认地址通常为http://localhost:8501。

## 深度学习原理简述

本应用的视觉提取核心由两个深度神经网络串联组成，具备极高的鲁棒性：

**文字检测网络 (基于 CRAFT 架构)**：

输入原图 RGB 矩阵，通过卷积神经网络（CNN）提取特征，输出每个像素“属于文字区域”的概率热力图，从而精准裁剪出单行文字的边界框（Bounding Boxes）。

**文字识别网络 (基于 CRNN 架构)**：

将裁剪出的单行图像送入网络，CNN 提取视觉特征后，利用双向 LSTM 捕捉字符间的上下文序列关系，最终通过 CTC（连接主义时序分类）解码器映射输出为高置信度的字符串。

## 项目结构

```plaintext
.
├── app.py               # Streamlit 主程序（包含 UI 渲染、模型加载与推理逻辑）
├── requirements.txt     # 项目依赖包清单
├── .gitignore           # Git 忽略配置（防止模型缓存上传）
└── README.md            # 项目说明文档
```

## 未来规划

虽然当前版本已具备完整的基础功能，但项目仍有广阔的迭代空间。后续计划探索以下方向：

- [ ] **本地大语言模型接入**：计划集成并调度本地 LLM（例如通过 Ollama 部署的模型），替代现有的 API 翻译，实现纯本地化、强隐私的离线工作流。
- [ ] **前后端分离架构重构**：为应对更复杂的并发和多智能体协同处理，考虑未来将底层推理服务迁移至 FastAPI，前端使用更灵活的框架（如 Vue 3）进行重构。
- [ ] **文档流批量处理**：增加对 PDF 文档和多图队列的支持，自动拼接与排版导出。

## 致谢与参考资料

本项目的顺利实现离不开以下优秀的开源社区与前沿研究：

- [EasyOCR](https://github.com/JaidedAI/EasyOCR): 感谢 JaidedAI 团队提供的轻量级、开箱即用的 OCR 引擎。
- [CRAFT-PyTorch](https://github.com/clovaai/CRAFT-pytorch): 强大的字符级边界框检测算法，保障了复杂排版下的高召回率。
- [deep-translator](https://github.com/nidhaloff/deep-translator): 稳定轻量的多语言翻译代理。

## 开源协议

本项目基于 [MIT License](LICENSE) 协议开源。允许任何人在保留原作者版权声明的前提下，自由使用、修改和分发本项目的代码。
