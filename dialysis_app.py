# # 1. 先激活环境（你已经做过）
# conda activate dialysis_py311

# # 2. 切换到脚本所在的文件夹（整条路径用双引号包住）
# cd "/Users/huaiwenchang/Desktop/FDU D.Sc./Aging/2023.5.18 Incremental Dialysis/Analysis/模型封装/shiny_app"

# # 3. 运行 Streamlit
# streamlit run dialysis_app.py

# ───────────── dialysis_app.py ─────────────────────────────────
"""
增 强 版：带单位显示 + 更美观的 Streamlit 页面
------------------------------------------------------------
• 作者：Fudan Univ. Incremental Dialysis Lab
• 说明：输入 28 项基线指标 → 预测患者是否需递增透析
• 依赖：streamlit 1.35, streamlit-option-menu, pandas, numpy, scikit-learn
"""

import pickle, pandas as pd
from pathlib import Path
import streamlit as st
from streamlit_option_menu import option_menu

# ========= 0. 可调整参数 =====================================================
THRESHOLD = 0.35              # 预测概率 ≥ THRESHOLD → 判为高风险
PRIMARY_COLOR = "#004c6d"     # 页面主色（深医疗蓝）
ACCENT_COLOR  = "#fac45a"     # 强调色（饱满琥珀）

# ========= 1. 路径、特征、单位 ==============================================
MODEL_DIR = Path(__file__).parent / "model"

FEATURES_UNITS = {
    # 透析时长 / 体征
    "Dialysis session length"        : "hour",
    "Pre-Dialysis Weight"            : "kg",
    "Pre-Dialysis SBP"               : "mmHg",
    "Pre-Dialysis DBP"               : "mmHg",
    "Pre-Dialysis Pulse"             : "bpm",
    "Post-Dialysis Weight"           : "kg",
    "Post-Dialysis SBP"              : "mmHg",
    "Post-Dialysis DBP"              : "mmHg",
    "Post-Dialysis Pulse"            : "bpm",
    # 机器参数
    "Total Blood Volume"             : "mL",
    "Total UV per session"           : "mL",
    "Ultrafiltration Rate"           : "mL /kg /h",
    "Ultrafiltration Weight Ratio"   : "%",
    "Mean Blood Flow"                : "mL/min",
    "Mean Arterial Pressure"         : "mmHg",
    "Mean Venous Pressure"           : "mmHg",
    "Mean TMP"                       : "mmHg",
    "Mean Dialysate Flow Rate"       : "mL/min",
    "Mean Dialysate Temperature"     : "°C",
    "Mean Conductivity"              : "mS/cm",
    # 生化
    "ProBNP"                         : "pg/mL",
    "Hemoglobin"                     : "g/dL",
    "Pre-Dialysis Creatinine"        : "µmol/L",
    "Total Bilirubin"                : "µmol/L",
    "Hs-CRP"                         : "mg/L",
    "GFR"                            : "mL/min/1.73 m²",
    "PTH"                            : "pg/mL",
    "nPCR"                           : "g/kg/day"
}
FEATURES = list(FEATURES_UNITS.keys())   # 顺序保持与训练一致

# ========= 2. 缓存读取模型 & scaler ==========================================
@st.cache_resource(show_spinner=False)
def load_artifacts():
    with open(MODEL_DIR / "best_model.pkl", "rb") as fm,\
         open(MODEL_DIR / "scaler.pkl",      "rb") as fs:
        return pickle.load(fm), pickle.load(fs)

model, scaler = load_artifacts()

# ========= 3. 页面基础设置与自定义 CSS =======================================
st.set_page_config("Incremental Dialysis Predictor", "🩺", layout="centered", initial_sidebar_state="collapsed")
st.markdown(f"""
<style>
/* 统一字体 & 颜色 */
body {{font-family:'Helvetica Neue',Arial,sans-serif;}}
.stApp h1 {{color:{PRIMARY_COLOR}; font-weight:700;}}
/* 按钮 */
div.stButton>button {{
    background:{PRIMARY_COLOR}; border:none; color:white; 
    font-size:16px; border-radius:8px; padding:0.5em 1.3em;
}}
/* 概率徽章 */
.prob-badge {{
    display:inline-block; background:{ACCENT_COLOR}; color:#000;
    padding:0.25em 0.7em; border-radius:10px; font-weight:600;
}}
</style>
""", unsafe_allow_html=True)

# ========= 4. 顶部导航栏 =====================================================
page = option_menu(
    None, ["预测", "关于"],
    icons=["activity","info-circle"],
    orientation="horizontal",
    styles={"nav-link-selected": {"background-color":"#e7eff6"}}
)

# ========= 5. 预测页 ==========================================================
if page == "预测":
    st.title("递增透析风险预测器")
    st.write("**请录入患者的生化及机器参数。** 以下默认值均为 0，可逐项修改。")

    # 二列表单
    col1, col2 = st.columns(2)
    user_input = {}
    with col1:
        for feat in FEATURES[:len(FEATURES)//2]:
            unit = FEATURES_UNITS[feat]
            user_input[feat] = st.number_input(f"{feat} ({unit})", value=0.0, step=0.01, format="%.2f")
    with col2:
        for feat in FEATURES[len(FEATURES)//2:]:
            unit = FEATURES_UNITS[feat]
            user_input[feat] = st.number_input(f"{feat} ({unit})", value=0.0, step=0.01, format="%.2f")

    # 运行按钮
    if st.button("开始预测"):
        X = pd.DataFrame([user_input])
        prob = model.predict_proba(scaler.transform(X))[:,1][0]
        pred = "高风险：建议递增透析频率" if prob >= THRESHOLD else "低风险：维持当前透析频率"
        
        st.divider()
        st.subheader("预测结果")
        st.markdown(
            f"<span class='prob-badge'>概率：{prob:.2%}</span>&nbsp;&nbsp;**{pred}**",
            unsafe_allow_html=True
        )
        with st.expander("⇢ 查看输入详情"):
            st.dataframe(X.style.format(precision=2))

# ========= 6. 关于页 ==========================================================
else:
    st.title("关于本工具")
    st.markdown(f"""
**Incremental Dialysis Predictor**（递增透析预测器）  
利用华山医院内部队列训练的机器学习模型，依据 28 项基线指标，预测患者在随访期间进入递增透析方案的概率。

| 模型信息 | 说明 |
|----------|------|
| 算法     | Extra Trees |
| 特征数   | {len(FEATURES)} |
| 前端框架 | Streamlit 1.35 |

> **仅供科研/教学使用，不可直接作为临床决策依据。**

----

© 2025 Fudan University • Huashan Hospital Chen Lab  
若有建议，请联系 `hwchang21@m.fudan.edu.cn`
""")
