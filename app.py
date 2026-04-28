"""
========================================================================
项目名称: 基于深度学习的课本 OCR 提取与翻译助手
作    者: [你的姓名]
技术栈 : Streamlit + EasyOCR (PyTorch) + deep-translator
========================================================================
"""

import streamlit as st
import easyocr
import numpy as np
from PIL import Image
from deep_translator import GoogleTranslator

# ──────────────────────────────────────────────
# 页面基础配置（必须是第一个 Streamlit 调用）
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="课本 OCR 翻译助手",
    page_icon="📖",
    layout="wide",
)

# ──────────────────────────────────────────────
# 全局样式注入
# ──────────────────────────────────────────────
st.markdown("""
<style>
/* ── 字体引入 ── */
@import url('https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;600;700&family=JetBrains+Mono:wght@400;500&family=Crimson+Pro:ital,wght@0,400;0,600;1,400&display=swap');

/* ── 全局变量 ── */
:root {
    --ink:        #1a1208;
    --parchment:  #fdf6e3;
    --sepia:      #c8a96e;
    --sepia-dark: #8b6914;
    --accent:     #b5451b;
    --muted:      #7a6a52;
    --paper:      #fffdf7;
    --rule:       #e8dcc8;
    --shadow:     rgba(139, 105, 20, 0.15);
}

/* ── 整体背景 ── */
.stApp {
    background-color: var(--parchment);
    background-image:
        repeating-linear-gradient(
            0deg,
            transparent,
            transparent 27px,
            rgba(200,169,110,0.18) 27px,
            rgba(200,169,110,0.18) 28px
        );
    font-family: 'Noto Serif SC', serif;
}

/* ── 顶部大标题区域 ── */
.hero-header {
    text-align: center;
    padding: 2.4rem 0 1.6rem;
    border-bottom: 2px solid var(--sepia);
    margin-bottom: 2rem;
}
.hero-title {
    font-family: 'Crimson Pro', 'Noto Serif SC', serif;
    font-size: 2.8rem;
    font-weight: 600;
    color: var(--ink);
    letter-spacing: 0.04em;
    margin: 0;
}
.hero-title span { color: var(--accent); }
.hero-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: var(--muted);
    letter-spacing: 0.12em;
    margin-top: 0.5rem;
    text-transform: uppercase;
}

/* ── 内容卡片 ── */
.card {
    background: var(--paper);
    border: 1px solid var(--rule);
    border-radius: 4px;
    padding: 1.6rem 1.8rem;
    box-shadow: 0 2px 12px var(--shadow);
    margin-bottom: 1.4rem;
}
.card-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--sepia-dark);
    border-bottom: 1px solid var(--rule);
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
}

/* ── 文本展示区 ── */
.text-display {
    font-family: 'Crimson Pro', serif;
    font-size: 1.12rem;
    line-height: 2;
    color: var(--ink);
    white-space: pre-wrap;
    word-break: break-word;
    min-height: 120px;
}
.text-display.zh {
    font-family: 'Noto Serif SC', serif;
    font-size: 1.05rem;
    color: var(--accent);
}

/* ── 按钮覆写 ── */
.stButton > button {
    background: var(--ink) !important;
    color: var(--parchment) !important;
    border: none !important;
    border-radius: 2px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.1em !important;
    padding: 0.6rem 2rem !important;
    transition: background 0.2s ease, transform 0.15s ease !important;
    width: 100%;
}
.stButton > button:hover {
    background: var(--accent) !important;
    transform: translateY(-1px) !important;
}

/* ── 侧边栏 ── */
[data-testid="stSidebar"] {
    background: #f5edda !important;
    border-right: 1px solid var(--rule) !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] p {
    font-family: 'Noto Serif SC', serif !important;
    color: var(--ink) !important;
}
.sidebar-section-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--muted);
    margin: 1.2rem 0 0.4rem;
    padding-bottom: 0.3rem;
    border-bottom: 1px solid var(--rule);
}

/* ── 角标徽章 ── */
.badge {
    display: inline-block;
    background: var(--sepia);
    color: var(--ink);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.1em;
    padding: 0.15rem 0.5rem;
    border-radius: 2px;
    vertical-align: middle;
    margin-left: 0.5rem;
}

/* ── 图片区域 ── */
[data-testid="stImage"] img {
    border: 1px solid var(--rule);
    border-radius: 2px;
    box-shadow: 0 4px 16px var(--shadow);
}

/* ── 隐藏默认菜单 ── */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# 模块一：OCR 读取器初始化
# ════════════════════════════════════════════════════════════════

@st.cache_resource  # Streamlit 缓存：整个应用生命周期只初始化一次模型，避免重复加载
def init_ocr_reader(languages: list) -> easyocr.Reader:
    """
    初始化 EasyOCR 读取器（深度学习模型加载）。

    【深度学习原理简述 - 写进实验报告】
    EasyOCR 的底层由两个深度神经网络串联组成：
    
    1. 文字检测网络（基于 CRAFT 架构）：
       - 输入：整张图片（RGB 像素矩阵）
       - 过程：卷积神经网络（CNN）逐层提取图像特征，
               输出每个像素"是文字区域"的概率热力图。
       - 输出：框出所有文字区域的边界框（Bounding Boxes）。
    
    2. 文字识别网络（基于 CRNN 架构）：
       - 输入：检测阶段裁剪出的单行文字图像
       - 过程：CNN 提取视觉特征 → 双向 LSTM 捕捉字符间的上下文序列关系
               → CTC（连接主义时序分类）解码器将序列映射成字符串。
       - 输出：识别出的文字字符串与置信度分数。
    
    gpu=False 参数：在普通 CPU 上运行推理。
    若有 NVIDIA GPU 并安装了 CUDA，将其改为 True 可大幅提速。
    """
    reader = easyocr.Reader(languages, gpu=False)
    return reader


# ════════════════════════════════════════════════════════════════
# 模块二：文本提取（OCR 推理）
# ════════════════════════════════════════════════════════════════

def extract_text(reader: easyocr.Reader, image: Image.Image) -> str:
    """
    调用 EasyOCR 对上传图片执行 OCR 推理，提取文本。

    参数:
        reader : 已初始化的 EasyOCR Reader 对象（含加载好的神经网络权重）
        image  : PIL Image 对象（用户上传的图片）

    返回:
        提取出的完整文本字符串；若未识别到文字，返回空字符串。

    【技术细节】
    - reader.readtext() 接受 numpy 数组格式的图像输入。
    - 返回值是一个列表，每个元素为 (边界框坐标, 识别文本, 置信度) 的三元组。
    - detail=1 保留置信度信息，detail=0 则只返回文本字符串列表。
    - 本函数过滤置信度 < 0.3 的结果，减少噪声干扰。
    """
    # 将 PIL Image 转为 numpy 数组，这是 EasyOCR 期望的输入格式
    img_array = np.array(image)

    # 执行深度学习推理（核心调用）
    # paragraph=True 会自动将邻近的文字行合并成段落，适合课本场景
    results = reader.readtext(img_array, paragraph=False)

    if not results:
        return ""

    # 拼接所有置信度达标的识别结果，用换行符分隔
    lines = []
    for (_, text, confidence) in results:
        if confidence >= 0.3:
            lines.append(text)

    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════
# 模块三：文本翻译
# ════════════════════════════════════════════════════════════════

def translate_text(text: str, source_lang: str) -> str:
    """
    使用 Google Translate API（通过 deep-translator 库）将提取文本翻译为中文。

    参数:
        text        : 待翻译的原始文本
        source_lang : 源语言代码，'de'（德语）或 'en'（英语）

    返回:
        翻译后的中文字符串。

    【技术说明】
    deep-translator 封装了 Google Translate 的免费网页接口，
    无需 API Key，适合学术项目使用。
    target 固定为 'zh-CN'（简体中文）。
    """
    translator = GoogleTranslator(source=source_lang, target='zh-CN')

    # Google Translate 单次请求有字符上限（约 5000 字符）
    # 对于课本段落，通常不会超限；若需处理长文，可分段翻译
    translated = translator.translate(text)
    return translated


# ════════════════════════════════════════════════════════════════
# UI 主体：侧边栏配置
# ════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 📖 OCR 翻译助手")
    st.markdown("---")

    st.markdown('<div class="sidebar-section-title">源语言设置</div>', unsafe_allow_html=True)
    lang_option = st.radio(
        "图片中的文字语言",
        options=["🇩🇪 德语 (German)", "🇬🇧 英语 (English)"],
        index=0,
        help="选择图片中文字的语言，将影响 OCR 模型的语言识别策略。"
    )

    # 将界面选项映射为 EasyOCR 和 deep-translator 的语言代码
    if "德语" in lang_option:
        ocr_langs = ['de', 'en']  # 德语文本中常混有英文，同时加载两种模型
        src_lang_code = 'de'
        lang_display = "德语"
    else:
        ocr_langs = ['en']
        src_lang_code = 'en'
        lang_display = "英语"

    st.markdown('<div class="sidebar-section-title">模型信息</div>', unsafe_allow_html=True)
    st.caption("**OCR 引擎**：EasyOCR (CRAFT + CRNN)")
    st.caption("**翻译服务**：Google Translate")
    st.caption("**推理设备**：CPU")

    st.markdown("---")
    st.markdown('<div class="sidebar-section-title">使用说明</div>', unsafe_allow_html=True)
    st.markdown("""
1. 选择源语言  
2. 上传课本截图  
3. 点击"开始提取与翻译"  
4. 左侧查看原文，右侧查看译文
""")


# ════════════════════════════════════════════════════════════════
# UI 主体：顶部标题
# ════════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero-header">
    <p class="hero-title">课本 OCR <span>提取与翻译</span>助手</p>
    <p class="hero-sub">Deep Learning · EasyOCR · Google Translate &nbsp;|&nbsp; 基于深度学习的自动化阅读辅助工具</p>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# UI 主体：图片上传与预览
# ════════════════════════════════════════════════════════════════

st.markdown('<div class="card"><div class="card-label">① 上传图片</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    label="支持 JPG / PNG / WEBP 格式，建议使用清晰的课本截图",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="visible"
)
st.markdown('</div>', unsafe_allow_html=True)

# 若有图片上传，显示预览
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.markdown('<div class="card"><div class="card-label">② 图片预览</div>', unsafe_allow_html=True)
    st.image(image, caption=f"已上传：{uploaded_file.name}", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── 触发按钮 ──
    st.markdown('<div class="card"><div class="card-label">③ 执行 AI 推理</div>', unsafe_allow_html=True)
    run_btn = st.button(f"🔍  开始提取与翻译（{lang_display} → 中文）")
    st.markdown('</div>', unsafe_allow_html=True)

    if run_btn:
        # ── 步骤一：加载/复用 OCR 模型（含进度提示）──
        with st.spinner("⚙️ 正在加载深度学习模型（首次运行需要下载权重，请耐心等待…）"):
            try:
                reader = init_ocr_reader(ocr_langs)
            except Exception as e:
                st.error(f"❌ 模型初始化失败，请检查网络连接或 EasyOCR 安装：\n\n`{e}`")
                st.stop()

        # ── 步骤二：OCR 推理 ──
        with st.spinner("🧠 AI 正在识别图片中的文字（CRAFT 检测 + CRNN 识别）…"):
            try:
                extracted_text = extract_text(reader, image)
            except Exception as e:
                st.error(f"❌ OCR 推理过程中发生错误：\n\n`{e}`")
                st.stop()

        # 处理未识别到文字的情况
        if not extracted_text.strip():
            st.warning("⚠️ 未能从图片中识别到有效文字。\n\n请确认：① 图片清晰度足够；② 选择了正确的源语言；③ 图片中包含可辨认的印刷体文字。")
            st.stop()

        # ── 步骤三：调用翻译 API ──
        with st.spinner("🌐 正在调用 Google Translate 翻译为中文…"):
            try:
                translated_text = translate_text(extracted_text, src_lang_code)
            except Exception as e:
                # 翻译失败时，仍展示 OCR 原文，不让整个流程中断
                st.warning(f"⚠️ 翻译服务暂时不可用（可能是网络超时）：`{e}`\n\n以下为 OCR 提取的原文：")
                translated_text = "（翻译失败，请检查网络连接后重试）"

        # ════════════════════════════════════════════════════════════
        # UI：双栏结果展示
        # ════════════════════════════════════════════════════════════
        st.markdown("---")
        st.markdown("#### 📄 识别与翻译结果")

        col_left, col_right = st.columns(2, gap="large")

        with col_left:
            st.markdown(f"""
<div class="card">
    <div class="card-label">提取原文 <span class="badge">{lang_display}</span></div>
    <div class="text-display">{extracted_text}</div>
</div>
""", unsafe_allow_html=True)

        with col_right:
            st.markdown(f"""
<div class="card">
    <div class="card-label">中文译文 <span class="badge">ZH-CN</span></div>
    <div class="text-display zh">{translated_text}</div>
</div>
""", unsafe_allow_html=True)

        # ── 可选：提供文本下载 ──
        combined_output = f"【原文（{lang_display}）】\n{extracted_text}\n\n【中文译文】\n{translated_text}"
        st.download_button(
            label="⬇️  下载对照文本（.txt）",
            data=combined_output.encode("utf-8"),
            file_name="ocr_translation_result.txt",
            mime="text/plain"
        )

else:
    # 未上传图片时，显示引导占位区
    st.markdown("""
<div class="card" style="text-align:center; padding: 3rem 2rem; color: var(--muted);">
    <div style="font-size:3rem; margin-bottom:1rem;">📷</div>
    <div style="font-family:'Crimson Pro',serif; font-size:1.3rem;">
        请在上方上传一张课本图片
    </div>
    <div style="font-family:'JetBrains Mono',monospace; font-size:0.72rem; margin-top:0.6rem; letter-spacing:0.08em;">
        支持德语 / 英语教材截图 · 清晰印刷体效果最佳
    </div>
</div>
""", unsafe_allow_html=True)
