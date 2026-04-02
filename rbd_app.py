import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# ================================
# إعداد الصفحة والواجهة
# ================================
st.set_page_config(page_title="Multi-Test Statistical Analysis Suite", page_icon="📊", layout="wide")

# تطبيق الثيم الداكن
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: white; }
    .stMarkdown, .stText, .stDataFrame { color: white; }
    .css-1aumxhk { color: white; }
</style>
""", unsafe_allow_html=True)

st.title("📊 Multi-Test Statistical Analysis Suite")
st.markdown("### *ANOVA (RBD) | Two‑Sample t‑test | Two‑Sample Z‑test | Tukey HSD Post‑hoc*")
st.markdown("---")

# الشريط الجانبي
with st.sidebar:
    st.header("⚙️ Settings")
    test_type = st.selectbox(
        "Select Statistical Test:",
        ["ANOVA (F-test) - RBD", "Two-Sample t-test", "Two-Sample Z-test", "Tukey HSD (Post-hoc)"]
    )
    data_source = st.radio(
        "Data input method:",
        ["✏️ Manual entry", "📂 Upload Excel/CSV"]
    )
    alpha = st.number_input("Significance level (α)", min_value=0.01, max_value=0.20, value=0.05, step=0.01)

# ================================
# دوال تحميل البيانات
# ================================
def load_data_for_anova():
    if data_source == "✏️ Manual entry":
        st.subheader("Manual Data Entry for RBD")
        col1, col2 = st.columns(2)
        with col1:
            t = st.number_input("Number of treatments", min_value=2, max_value=10, value=3)
        with col2:
            b = st.number_input("Number of blocks", min_value=2, max_value=10, value=4)
        treatment_names = []
        block_names = []
        st.write("**Treatment names:**")
        cols = st.columns(int(t))
        for i in range(int(t)):
            with cols[i]:
                name = st.text_input(f"Treatment {i+1}", value=f"T{i+1}")
                treatment_names.append(name)
        st.write("**Block names:**")
        cols = st.columns(int(b))
        for j in range(int(b)):
            with cols[j]:
                name = st.text_input(f"Block {j+1}", value=f"B{j+1}")
                block_names.append(name)
        data_input = []
        st.write("**Response values:**")
        for i in range(int(t)):
            cols = st.columns(int(b))
            row = []
            for j in range(int(b)):
                with cols[j]:
                    val = st.number_input(f"{treatment_names[i]}, {block_names[j]}", value=10.0, key=f"cell_{i}_{j}")
                    row.append(val)
            data_input.append(row)
        Y = np.array(data_input)
        return Y, treatment_names, block_names
    else:  # Upload file
        st.subheader("Upload Data File for RBD")
        uploaded = st.file_uploader("Choose Excel or CSV file", type=['xlsx', 'csv'])
        if uploaded:
            if uploaded.name.endswith('.csv'):
                df = pd.read_csv(uploaded, index_col=0)
            else:
                df = pd.read_excel(uploaded, index_col=0)
            st.write("Data preview:")
            st.dataframe(df, use_container_width=True)
            return df.values, df.index.tolist(), df.columns.tolist()
    return None, None, None

def load_data_for_two_groups():
    if data_source == "✏️ Manual entry":
        st.subheader("Manual Data Entry for Two Groups")
        g1_str = st.text_area("Group 1 values (comma separated)", "23,25,28,22,26")
        g2_str = st.text_area("Group 2 values (comma separated)", "19,21,24,20,22")
        try:
            g1 = np.array([float(x.strip()) for x in g1_str.split(',')])
            g2 = np.array([float(x.strip()) for x in g2_str.split(',')])
            return g1, g2, ["Group 1", "Group 2"]
        except:
            st.error("Invalid numbers. Use commas.")
            return None, None, None
    else:
        st.subheader("Upload Data File for Two Groups")
        uploaded = st.file_uploader("Choose Excel or CSV", type=['xlsx', 'csv'])
        if uploaded:
            if uploaded.name.endswith('.csv'):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded)
            if df.shape[1] >= 2:
                st.dataframe(df, use_container_width=True)
                return df.iloc[:,0].values, df.iloc[:,1].values, df.columns[:2].tolist()
            else:
                st.error("File must have at least two columns.")
    return None, None, None

def load_data_for_tukey():
    if data_source == "✏️ Manual entry":
        st.subheader("Manual Data Entry for Tukey HSD")
        st.write("Enter data for multiple groups, one group per line: GroupName: val1,val2,...")
        text_input = st.text_area("Groups data", "Group1: 23,25,28\nGroup2: 19,21,24\nGroup3: 15,18,20")
        values = []
        labels = []
        try:
            for line in text_input.strip().split('\n'):
                if ':' in line:
                    label, vals_str = line.split(':')
                    vals = [float(x.strip()) for x in vals_str.split(',')]
                    values.extend(vals)
                    labels.extend([label.strip()] * len(vals))
            return np.array(values), np.array(labels)
        except:
            st.error("Error parsing input.")
            return None, None
    else:
        st.subheader("Upload Data File for Tukey HSD")
        st.info("Upload a file where each column represents a group.")
        uploaded = st.file_uploader("Choose file", type=['xlsx', 'csv'])
        if uploaded:
            if uploaded.name.endswith('.csv'):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded)
            st.dataframe(df, use_container_width=True)
            values = []
            labels = []
            for col in df.columns:
                vals = df[col].dropna().values
                values.extend(vals)
                labels.extend([col] * len(vals))
            return np.array(values), np.array(labels)
    return None, None

# ================================
# زر التحليل الرئيسي
# ================================
if st.button("🔬 Run Analysis", type="primary"):
    if test_type == "ANOVA (F-test) - RBD":
        data = load_data_for_anova()
        if data[0] is None:
            st.stop()
        Y, treatment_names, block_names = data
        t, b = Y.shape
        grand_mean = np.mean(Y)
        treatment_means = np.mean(Y, axis=1)
        block_means = np.mean(Y, axis=0)

        SST = np.sum((Y - grand_mean)**2)
        SSTreat = b * np.sum((treatment_means - grand_mean)**2)
        SSBlock = t * np.sum((block_means - grand_mean)**2)
        SSE = SST - SSTreat - SSBlock

        df_treat = t - 1
        df_block = b - 1
        df_error = (t-1)*(b-1)
        df_total = t*b - 1

        MSTreat = SSTreat / df_treat
        MSBlock = SSBlock / df_block
        MSE = SSE / df_error

        F_treat = MSTreat / MSE
        F_block = MSBlock / MSE

        F_crit_treat = stats.f.ppf(1-alpha, df_treat, df_error)
        F_crit_block = stats.f.ppf(1-alpha, df_block, df_error)
        p_treat = 1 - stats.f.cdf(F_treat, df_treat, df_error)
        p_block = 1 - stats.f.cdf(F_block, df_block, df_error)

        st.markdown("---")
        st.header("📈 ANOVA Results")
        anova_df = pd.DataFrame({
            'Source': ['Treatments', 'Blocks', 'Error', 'Total'],
            'SS': [SSTreat, SSBlock, SSE, SST],
            'df': [df_treat, df_block, df_error, df_total],
            'MS': [MSTreat, MSBlock, MSE, np.nan],
            'F': [F_treat, F_block, np.nan, np.nan],
            'p-value': [p_treat, p_block, np.nan, np.nan]
        })
        anova_display = anova_df.round(2)
        anova_display['p-value'] = anova_display['p-value'].round(4)
        anova_display = anova_display.fillna('')
        st.dataframe(anova_display, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**Grand Mean:** {grand_mean:.2f}")
            st.write("**Treatment Means:**")
            for name, val in zip(treatment_names, treatment_means):
                st.write(f"- {name}: {val:.2f}")
        with col2:
            st.success(f"**F (Treatments):** {F_treat:.2f}")
            st.write(f"**Critical F:** {F_crit_treat:.2f}")
            st.write(f"**p-value:** {p_treat:.4f}")
            if p_treat < alpha:
                st.error("❌ **Reject H0:** Significant differences among treatments.")
                show_tukey = True
            else:
                st.success("✅ **Fail to reject H0:** No significant differences.")
                show_tukey = False
        with col3:
            st.success(f"**F (Blocks):** {F_block:.2f}")
            st.write(f"**Critical F:** {F_crit_block:.2f}")
            st.write(f"**p-value:** {p_block:.4f}")
            if p_block < alpha:
                st.warning("⚠️ **Significant block effect.**")
            else:
                st.info("✅ **No significant block effect.**")

        # Bar plots
        fig, axes = plt.subplots(1, 2, figsize=(12,4))
        axes[0].bar(treatment_names, treatment_means, color='skyblue', edgecolor='navy')
        axes[0].set_xlabel('Treatments')
        axes[0].set_ylabel('Mean Response')
        axes[0].set_title('Treatment Means')
        axes[0].grid(axis='y', linestyle='--', alpha=0.7)
        axes[1].bar(block_names, block_means, color='lightcoral', edgecolor='darkred')
        axes[1].set_xlabel('Blocks')
        axes[1].set_ylabel('Mean Response')
        axes[1].set_title('Block Means')
        axes[1].grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        st.pyplot(fig)

        if show_tukey:
            with st.expander("🔍 Tukey HSD Post-hoc Test"):
                long_data = []
                for i, treat in enumerate(treatment_names):
                    for j in range(b):
                        long_data.append((Y[i,j], treat))
                df_long = pd.DataFrame(long_data, columns=['value', 'treatment'])
                tukey_res = pairwise_tukeyhsd(df_long['value'], df_long['treatment'], alpha=alpha)
                st.dataframe(pd.DataFrame(data=tukey_res.summary().data[1:], columns=tukey_res.summary().data[0]))
                fig_t, ax_t = plt.subplots()
                tukey_res.plot_simultaneous(ax=ax_t)
                st.pyplot(fig_t)

        with st.expander("Show Operating Characteristic (OC) Curve"):
            lambda_vals = np.linspace(0, 30, 200)
            beta_vals = [stats.ncf.cdf(F_crit_treat, df_treat, df_error, lam) for lam in lambda_vals]
            fig_oc, ax_oc = plt.subplots(figsize=(8,4))
            ax_oc.plot(lambda_vals, beta_vals, 'b-', linewidth=2)
            ax_oc.axhline(y=1-alpha, color='r', linestyle='--', label=f'β = {1-alpha}')
            ax_oc.set_xlabel('Noncentrality parameter λ')
            ax_oc.set_ylabel('Probability of accepting H₀ (β)')
            ax_oc.set_title('Operating Characteristic Curve')
            ax_oc.grid(True, linestyle='--', alpha=0.5)
            ax_oc.legend()
            st.pyplot(fig_oc)

    elif test_type == "Two-Sample t-test":
        g1, g2, group_names = load_data_for_two_groups()
        if g1 is None:
            st.stop()
        t_stat, p_val = stats.ttest_ind(g1, g2, equal_var=False)
        df_welch = (np.var(g1, ddof=1)/len(g1) + np.var(g2, ddof=1)/len(g2))**2 / (
            (np.var(g1, ddof=1)/len(g1))**2/(len(g1)-1) + (np.var(g2, ddof=1)/len(g2))**2/(len(g2)-1)
        )
        crit_t = stats.t.ppf(1-alpha/2, df_welch)
        st.markdown("---")
        st.header("📈 Two-Sample t-test Results (Welch)")
        st.write(f"**{group_names[0]}:** n={len(g1)}, mean={np.mean(g1):.2f}, std={np.std(g1, ddof=1):.2f}")
        st.write(f"**{group_names[1]}:** n={len(g2)}, mean={np.mean(g2):.2f}, std={np.std(g2, ddof=1):.2f}")
        st.write(f"**t-statistic:** {t_stat:.4f}")
        st.write(f"**Degrees of freedom (Welch):** {df_welch:.2f}")
        st.write(f"**Critical t (α={alpha}):** ±{crit_t:.4f}")
        st.write(f"**p-value:** {p_val:.4f}")
        if p_val < alpha:
            st.error("❌ Reject H0: Significant difference.")
        else:
            st.success("✅ Fail to reject H0: No significant difference.")
        fig, ax = plt.subplots()
        ax.boxplot([g1, g2], labels=group_names)
        ax.set_ylabel('Values')
        st.pyplot(fig)

    elif test_type == "Two-Sample Z-test":
        g1, g2, group_names = load_data_for_two_groups()
        if g1 is None:
            st.stop()
        var1 = np.var(g1, ddof=1)
        var2 = np.var(g2, ddof=1)
        se = np.sqrt(var1/len(g1) + var2/len(g2))
        z_stat = (np.mean(g1) - np.mean(g2)) / se
        p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        crit_z = stats.norm.ppf(1 - alpha/2)
        st.markdown("---")
        st.header("📈 Two-Sample Z-test Results")
        st.write(f"**{group_names[0]}:** mean={np.mean(g1):.2f}, n={len(g1)}")
        st.write(f"**{group_names[1]}:** mean={np.mean(g2):.2f}, n={len(g2)}")
        st.write(f"**Z-statistic:** {z_stat:.4f}")
        st.write(f"**Critical Z (α={alpha}):** ±{crit_z:.4f}")
        st.write(f"**p-value:** {p_val:.4f}")
        if p_val < alpha:
            st.error("❌ Reject H0: Significant difference.")
        else:
            st.success("✅ Fail to reject H0: No significant difference.")
        fig, ax = plt.subplots()
        ax.boxplot([g1, g2], labels=group_names)
        st.pyplot(fig)

    elif test_type == "Tukey HSD (Post-hoc)":
        values, labels = load_data_for_tukey()
        if values is None:
            st.stop()
        tukey_res = pairwise_tukeyhsd(values, labels, alpha=alpha)
        st.markdown("---")
        st.header("📈 Tukey HSD Results")
        st.dataframe(pd.DataFrame(data=tukey_res.summary().data[1:], columns=tukey_res.summary().data[0]))
        fig, ax = plt.subplots()
        tukey_res.plot_simultaneous(ax=ax)
        st.pyplot(fig)

    st.success("✅ Analysis complete!")

st.markdown("---")
st.caption("Multi-Test Statistical Analysis Suite – Manual entry or Excel/CSV upload only.")