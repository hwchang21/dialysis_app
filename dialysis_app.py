# ======= 基础库 =======
import pickle
import pandas as pd
from pathlib import Path
import streamlit as st
from streamlit_option_menu import option_menu


# ======= 0. 可自定义参数 =====================================
PRIMARY  = "#004c6d"   # 页面主色
ACCENT   = "#fac45a"   # 强调色
DEFAULT_THRESHOLD = 0.35     # 默认阈值

# ── 推荐阈值 & 文字说明（源于补充表 S2，已口语化） ───────────
THRESHOLD_TIPS = {
    0.15: "几乎不漏检，但随访负担大",
    0.35: "最优平衡推荐（默认）",
    0.50: "可能漏掉部分高风险患者",
    0.65: "仅适用于高度保守的预筛"
}

# ======= 1. 路径 / 特征列表 / 单位 ============================
MODEL_DIR = Path(__file__).parent / "model"

FEATURES_UNITS = {
    # 时长 & 体征
    "Dialysis session length"      : "h",
    "Pre-Dialysis Weight"          : "kg",
    "Pre-Dialysis SBP"             : "mmHg",
    "Pre-Dialysis DBP"             : "mmHg",
    "Pre-Dialysis Pulse"           : "bpm",
    "Post-Dialysis Weight"         : "kg",
    "Post-Dialysis SBP"            : "mmHg",
    "Post-Dialysis DBP"            : "mmHg",
    "Post-Dialysis Pulse"          : "bpm",
    # 机器参数
    "Total Blood Volume"           : "L",
    "Total UV per session"         : "mL",
    "Ultrafiltration Rate"         : "mL/h",
    "Ultrafiltration Weight Ratio" : "%",
    "Mean Blood Flow"              : "mL/min",
    "Mean Arterial Pressure"       : "mmHg",
    "Mean Venous Pressure"         : "mmHg",
    "Mean TMP"                     : "mmHg",
    "Mean Dialysate Flow Rate"     : "mL/min",
    "Mean Dialysate Temperature"   : "°C",
    "Mean Conductivity"            : "mS/cm",
    # 生化
    "ProBNP"                       : "pg/mL",
    "Hemoglobin"                   : "g/L",
    "Pre-Dialysis Creatinine"      : "µmol/L",
    "Total Bilirubin"              : "µmol/L",
    "Hs-CRP"                       : "mg/L",
    "GFR"                          : "mL/min/1.73 m²",
    "PTH"                          : "pg/mL",
    "nPCR"                         : "g/kg/day"
}
FEATURES = list(FEATURES_UNITS.keys())  # 顺序 = 训练顺序

# ======= 2. 读入模型 / 标准化器 (缓存) ========================
@st.cache_resource(show_spinner=False)
def load_artifacts():
    with open(MODEL_DIR / "best_model.pkl", "rb") as fm, \
         open(MODEL_DIR / "scaler.pkl",      "rb") as fs:
        return pickle.load(fm), pickle.load(fs)

model, scaler = load_artifacts()

# ======= 3. 页面基本设置 & CSS ================================
st.set_page_config("Incremental Dialysis Predictor", "🩺",
                   layout="centered", initial_sidebar_state="collapsed")
st.markdown(f"""
<style>
body {{font-family:'Helvetica Neue',Arial,sans-serif;}}
.stApp h1 {{color:{PRIMARY}; font-weight:700;}}
div.stButton>button {{
    background:{PRIMARY}; border:none; color:#fff;
    font-size:16px; border-radius:8px; padding:0.55em 1.4em;
}}
.prob-badge {{
    display:inline-block; background:{ACCENT}; color:#000;
    padding:0.3em 0.8em; border-radius:10px; font-weight:600;
}}
</style>
""", unsafe_allow_html=True)

# ======= 4. 侧边栏：阈值滑块 + 推荐表格 ======================
with st.sidebar:
    st.header("阈值（Probability Cut-off）")
    THRESHOLD = st.slider("判定为高风险的最低概率", 0.10, 0.75,
                          value=DEFAULT_THRESHOLD, step=0.05)
    # 推荐阈值表（阈值保留 2 位小数）
    tips_df = pd.DataFrame({
        "阈值": [f"{k:.3f}" for k in THRESHOLD_TIPS.keys()],
        "说明": [THRESHOLD_TIPS[k] for k in THRESHOLD_TIPS]
        })
    st.table(tips_df)
    st.caption("阈值推荐来自华山血透团队临床研究")

# ======= 5. 顶部导航 =========================================
page = option_menu(None, ["预测", "关于"],
                   icons=["activity", "info-circle"],
                   orientation="horizontal",
                   styles={"nav-link-selected":{"background-color":"#e7eff6"}})

# ------------------------------------------------------------------
# ▍ 预测页
# ------------------------------------------------------------------
if page == "预测":
    st.title("递增透析风险预测器")
    st.write("请录入患者 **28 项指标**（默认 0，可修改）")

    # ——输入表单（双列）——
    user_in = {}
    col1, col2 = st.columns(2)
    half = len(FEATURES) // 2
    for i, feat in enumerate(FEATURES):
        unit = FEATURES_UNITS[feat]
        container = col1 if i < half else col2
        with container:
            user_in[feat] = st.number_input(f"{feat} ({unit})",
                                            value=0.0, step=0.01,
                                            format="%.2f",
                                            key=f"input_{i}")

    # ——预测按钮——
    if st.button("运行预测"):
        X = pd.DataFrame([user_in])
        prob = float(model.predict_proba(scaler.transform(X))[:, 1][0])
        label = ("高风险：建议递增透析"
                 if prob >= THRESHOLD
                 else "低风险：维持当前透析方案")

        st.divider()
        st.subheader("预测结果")
        st.markdown(
            f"<span class='prob-badge'>概率：{prob:.2%}</span>&nbsp;&nbsp;**{label}**",
            unsafe_allow_html=True)

        with st.expander("输入参数明细"):
            st.dataframe(X.style.format(precision=2), use_container_width=True)

# ------------------------------------------------------------------
# ▍ 关于页
# ------------------------------------------------------------------
else:
    st.title("关于本工具")
    st.markdown(f"""
**Incremental Dialysis Predictor**（递增透析预测器）  
利用华山医院内部队列训练，预测患者需要递增每周透析频率的概率。 
 
| 模型信息 | 说明 |
|----------|------|
| 算法     | Extra Trees |
| 特征数   | {len(FEATURES)} |
| 前端框架 | Streamlit 1.35 |

> 仅供科研 / 教学参考，不可直接作为临床决策依据。

----
© 2025 Fudan University · Huashan Hospital • Chen Lab  
""")
