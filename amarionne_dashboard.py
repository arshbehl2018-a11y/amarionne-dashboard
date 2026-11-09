
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, mean_absolute_error, r2_score)
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(page_title="Amarionne | Dubai Feasibility", page_icon="🌿",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
[data-testid="stMetric"] {background:#f5f6f2; border:1px solid #e5e8df;
 padding:14px 16px; border-radius:12px;}
.hero {padding:22px 24px; border-radius:16px; background:linear-gradient(110deg,#203b32,#52705d);
 color:white; margin-bottom:18px;}
.hero h1 {color:white; margin:0; font-size:2.0rem;}
.hero p {color:#e7eee8; margin:8px 0 0 0;}
.note {padding:12px 16px; border-left:4px solid #b28b54; background:#faf7f0;
 border-radius:6px; color:#343a34;}
</style>
""", unsafe_allow_html=True)

OFFER_NAMES = {
    "Offer_1":"6-foot artificial trees with pots",
    "Offer_2":"Small/medium artificial plants",
    "Offer_3":"Premium artificial flowers",
    "Offer_4":"Plant styling consultation",
    "Offer_5":"Delivery and in-home placement",
    "Offer_6":"Wooden chopping/serving boards",
    "Offer_7":"Wooden trays/table accessories",
    "Offer_8":"Wooden mirrors/wall décor",
    "Offer_9":"Watch boxes/decorative storage",
    "Offer_10":"Complete apartment décor package"
}

@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return pd.read_csv("Amarionne_Synthetic_Customer_Data.csv")

def clean_data(raw):
    d = raw.copy()
    before = len(d)
    d = d.drop_duplicates().copy()
    duplicates = before - len(d)
    for c in ["Age","Willingness_to_Spend_AED","Interested_YesNo","Number_of_Offers_Interested"]:
        if c in d:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    for c in [x for x in d.columns if x.startswith("Offer_")]:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0).clip(0,1).astype(int)
    return d, duplicates

def prep_pipeline(X):
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    transformers = []
    if numeric:
        transformers.append(("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler())
        ]), numeric))
    if categorical:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical))
    return ColumnTransformer(transformers)

st.sidebar.title("🌿 Amarionne")
st.sidebar.caption("Dubai market-entry intelligence")
uploaded = st.sidebar.file_uploader("Upload survey CSV", type=["csv"])
try:
    raw = load_data(uploaded)
except Exception as e:
    st.error("Could not load the dataset. Put Amarionne_Synthetic_Customer_Data.csv in the same folder as this app, or upload a CSV.")
    st.exception(e)
    st.stop()

df, duplicate_count = clean_data(raw)
required = ["Interested_YesNo","Willingness_to_Spend_AED","Customer_Type","Dubai_Area"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Dataset is missing required columns: {missing}")
    st.stop()

# Filters
st.sidebar.subheader("Dashboard filters")
if "Customer_Type" in df:
    type_options = sorted(df.Customer_Type.dropna().astype(str).unique())
    selected_types = st.sidebar.multiselect("Customer type", type_options, default=type_options)
else: selected_types = []
if "Dubai_Area" in df:
    area_options = sorted(df.Dubai_Area.dropna().astype(str).unique())
    selected_areas = st.sidebar.multiselect("Dubai area", area_options, default=area_options)
else: selected_areas = []
d = df[df.Customer_Type.astype(str).isin(selected_types) & df.Dubai_Area.astype(str).isin(selected_areas)].copy()
if d.empty:
    st.warning("No records match the selected filters.")
    st.stop()

st.markdown("""
<div class="hero">
<h1>AMARIONNE INTERNATIONAL</h1>
<p>Dubai market feasibility dashboard · Customer intelligence · Product strategy · Financial scenarios</p>
</div>
""", unsafe_allow_html=True)
st.markdown('<div class="note"><b>Research integrity:</b> The supplied dataset is synthetic. Results demonstrate the analytics workflow and are not evidence of actual Dubai demand. Replace with real survey responses before making investment decisions.</div>', unsafe_allow_html=True)

# KPI row
interest = pd.to_numeric(d.Interested_YesNo, errors="coerce")
spend = pd.to_numeric(d.Willingness_to_Spend_AED, errors="coerce")
offer_cols = [c for c in OFFER_NAMES if c in d.columns]
k1,k2,k3,k4,k5 = st.columns(5)
k1.metric("Respondents", f"{len(d):,}")
k2.metric("Stated interest rate", f"{interest.mean()*100:.1f}%")
k3.metric("Average stated spend", f"AED {spend.mean():,.0f}")
k4.metric("Median stated spend", f"AED {spend.median():,.0f}")
k5.metric("Avg. offers selected", f"{d[offer_cols].sum(axis=1).mean():.1f}/10" if offer_cols else "N/A")

tabs = st.tabs(["Executive overview","Customer segments","Purchase prediction","Product combinations","Spending & feasibility","Data quality"])

with tabs[0]:
    st.subheader("Market response at a glance")
    a,b = st.columns(2)
    with a:
        interest_by = d.groupby("Customer_Type", dropna=False).Interested_YesNo.mean().mul(100).reset_index(name="Interest rate (%)")
        st.plotly_chart(px.bar(interest_by, x="Customer_Type", y="Interest rate (%)",
                               title="Stated interest by customer type", text_auto=".1f"), use_container_width=True)
    with b:
        if offer_cols:
            counts = d[offer_cols].sum().rename(index=OFFER_NAMES).sort_values().reset_index()
            counts.columns = ["Product/service","Interested respondents"]
            st.plotly_chart(px.bar(counts, x="Interested respondents", y="Product/service",
                                   orientation="h", title="Most selected offerings"), use_container_width=True)
    st.plotly_chart(px.box(d, x="Customer_Type", y="Willingness_to_Spend_AED",
                           color="Customer_Type", title="Stated spend distribution by customer type"),
                    use_container_width=True)
    st.caption("Interpretation: compare interest and spend together. A segment with high interest but low spending may require a lower-priced offer; a smaller segment with higher spend may be attractive for B2B packages.")

with tabs[1]:
    st.subheader("K-Means customer segmentation")
    cluster_cols = [c for c in ["Age","Sustainability_Importance_1_5","Quality_Importance_1_5",
                                "Price_Sensitivity_1_5","Number_of_Offers_Interested",
                                "Willingness_to_Spend_AED"] if c in d.columns]
    if len(cluster_cols) >= 2:
        k = st.slider("Number of clusters", 2, 6, 3)
        z = d[cluster_cols].apply(pd.to_numeric, errors="coerce")
        z = z.fillna(z.median()).fillna(0)
        scaled = StandardScaler().fit_transform(z)
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        d["Cluster"] = km.fit_predict(scaled).astype(str)
        summary = d.groupby("Cluster").agg(
            Customers=("Respondent_ID","count") if "Respondent_ID" in d else ("Interested_YesNo","size"),
            Interest_Rate=("Interested_YesNo","mean"),
            Avg_Spend=("Willingness_to_Spend_AED","mean"),
            Avg_Offers=("Number_of_Offers_Interested","mean") if "Number_of_Offers_Interested" in d else ("Interested_YesNo","mean")
        ).reset_index()
        summary["Interest_Rate"] *= 100
        st.dataframe(summary.style.format({"Interest_Rate":"{:.1f}%","Avg_Spend":"AED {:,.0f}","Avg_Offers":"{:.1f}"}), use_container_width=True)
        c1,c2 = st.columns(2)
        with c1:
            st.plotly_chart(px.scatter(d, x="Willingness_to_Spend_AED", y="Number_of_Offers_Interested",
                                       color="Cluster", hover_data=["Customer_Type"] if "Customer_Type" in d else None,
                                       title="Customer clusters: spend vs. offer interest"), use_container_width=True)
        with c2:
            st.plotly_chart(px.bar(summary, x="Cluster", y="Interest_Rate",
                                   title="Interest rate by discovered cluster"), use_container_width=True)
        st.info("Clusters are discovered statistically. Review their profiles before naming them low-, medium- or high-propensity segments.")
    else:
        st.warning("Not enough numeric profile variables to run clustering.")

with tabs[2]:
    st.subheader("Classification: who is likely to be interested?")
    st.caption("Target: Interested_YesNo. The self-reported Interest_Likelihood_1_5 is excluded to avoid target leakage.")
    feature_candidates = ["Age","Customer_Type","Dubai_Area","Property_Type","Monthly_Income_AED",
        "Decision_Role","Purchase_Timeline","Discovery_Channel","Preferred_Interior_Style",
        "Sustainability_Importance_1_5","Quality_Importance_1_5","Price_Sensitivity_1_5",
        "Home_Decor_Purchase_Frequency"]
    features = [c for c in feature_candidates if c in d.columns]
    if len(features) >= 2 and d.Interested_YesNo.nunique() == 2:
        X = d[features]
        y = d.Interested_YesNo.astype(int)
        pre = prep_pipeline(X)
        clf = Pipeline([("prep",pre),("model",RandomForestClassifier(
            n_estimators=180, random_state=42, class_weight="balanced", min_samples_leaf=3))])
        Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=.25,random_state=42,stratify=y)
        clf.fit(Xtr,ytr)
        pred = clf.predict(Xte)
        probs = clf.predict_proba(Xte)[:,1]
        m1,m2,m3,m4,m5 = st.columns(5)
        m1.metric("Accuracy",f"{accuracy_score(yte,pred):.1%}")
        m2.metric("Precision",f"{precision_score(yte,pred,zero_division=0):.1%}")
        m3.metric("Recall",f"{recall_score(yte,pred,zero_division=0):.1%}")
        m4.metric("F1 score",f"{f1_score(yte,pred,zero_division=0):.3f}")
        m5.metric("ROC-AUC",f"{roc_auc_score(yte,probs):.3f}")
        st.plotly_chart(px.histogram(x=probs, color=yte.astype(str), nbins=20,
                                     labels={"x":"Predicted probability","color":"Actual interest"},
                                     title="Predicted purchase-interest probabilities"), use_container_width=True)
        st.caption("Metrics are measured on a holdout sample. Synthetic data can produce misleadingly strong scores; validate on real responses.")
    else:
        st.warning("Classification needs both target classes and sufficient predictor columns.")

with tabs[3]:
    st.subheader("Association rules: what gets selected together?")
    if len(offer_cols) >= 2:
        basket = d[offer_cols].fillna(0).astype(bool)
        support = st.slider("Minimum support", 0.01, 0.40, 0.05, 0.01)
        confidence = st.slider("Minimum confidence", 0.10, 0.90, 0.30, 0.05)
        frequent = apriori(basket, min_support=support, use_colnames=True)
        if not frequent.empty:
            try:
                rules = association_rules(frequent, metric="confidence", min_threshold=confidence, num_itemsets=len(basket))
            except TypeError:
                rules = association_rules(frequent, metric="confidence", min_threshold=confidence)
            if not rules.empty:
                rules["Antecedent"] = rules.antecedents.apply(lambda s: " + ".join(OFFER_NAMES.get(x,x) for x in s))
                rules["Consequent"] = rules.consequents.apply(lambda s: " + ".join(OFFER_NAMES.get(x,x) for x in s))
                show = rules[["Antecedent","Consequent","support","confidence","lift"]].sort_values("lift",ascending=False)
                st.dataframe(show.head(25).style.format({"support":"{:.1%}","confidence":"{:.1%}","lift":"{:.2f}"}), use_container_width=True)
                st.caption("Prioritise rules with meaningful support and lift above 1. Confidence alone can be misleading when an item is already popular.")
            else: st.info("No rules meet these thresholds. Lower minimum support or confidence.")
        else: st.info("No frequent itemsets found at this support threshold.")
    else:
        st.warning("Dataset needs Offer_1 through Offer_10 binary columns.")

with tabs[4]:
    st.subheader("Spend modelling and commercial scenario")
    reg_features = [c for c in ["Age","Customer_Type","Property_Type","Monthly_Income_AED",
        "Decision_Role","Purchase_Timeline","Quality_Importance_1_5","Price_Sensitivity_1_5",
        "Number_of_Offers_Interested"] if c in d.columns]
    if len(reg_features) >= 2:
        X = d[reg_features]
        y = pd.to_numeric(d.Willingness_to_Spend_AED,errors="coerce")
        valid = y.notna()
        X,y = X.loc[valid],y.loc[valid]
        pre = prep_pipeline(X)
        reg = Pipeline([("prep",pre),("model",RandomForestRegressor(n_estimators=180,random_state=42,min_samples_leaf=3))])
        Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=.25,random_state=42)
        reg.fit(Xtr,ytr)
        p = reg.predict(Xte)
        q1,q2 = st.columns(2)
        q1.metric("Mean absolute error",f"AED {mean_absolute_error(yte,p):,.0f}")
        q2.metric("R²",f"{r2_score(yte,p):.3f}")
        st.caption("Regression estimates stated willingness to spend, not verified purchase value.")
    st.divider()
    st.markdown("#### Unit economics and break-even scenario")
    x1,x2,x3 = st.columns(3)
    selling = x1.number_input("Average selling price per order (AED)", min_value=1.0, value=299.0, step=10.0)
    landed = x2.number_input("Average landed cost per order (AED)", min_value=0.0, value=140.0, step=5.0)
    commission_pct = x3.number_input("Referral commission (%)", min_value=0.0, max_value=50.0, value=10.0, step=1.0)
    x4,x5 = st.columns(2)
    monthly_fixed = x4.number_input("Monthly fixed costs (AED)", min_value=0.0, value=5000.0, step=500.0)
    monthly_orders = x5.number_input("Expected monthly orders", min_value=0, value=60, step=5)
    contribution = selling - landed - selling*commission_pct/100
    break_even = (monthly_fixed / contribution) if contribution > 0 else np.inf
    revenue = selling * monthly_orders
    monthly_profit = contribution * monthly_orders - monthly_fixed
    z1,z2,z3,z4 = st.columns(4)
    z1.metric("Contribution per order",f"AED {contribution:,.2f}")
    z2.metric("Break-even orders/month", "Not viable" if not np.isfinite(break_even) else f"{np.ceil(break_even):,.0f}")
    z3.metric("Scenario monthly revenue",f"AED {revenue:,.0f}")
    z4.metric("Scenario monthly operating result",f"AED {monthly_profit:,.0f}",delta="Before tax and excluded costs")
    if contribution <= 0:
        st.error("This scenario has zero or negative contribution per order. Adjust price, cost or commission.")
    elif monthly_profit >= 0:
        st.success("This scenario covers the entered fixed costs. Validate every input with supplier and freight quotations.")
    else:
        st.warning("This scenario does not yet cover the entered monthly fixed costs.")
    st.caption("This is a simplified scenario calculator, not a full profit forecast. Include payment fees, returns, storage, installation, marketing, licensing, VAT treatment and other costs before investment decisions.")

with tabs[5]:
    st.subheader("Data quality and dataset profile")
    qc1,qc2,qc3 = st.columns(3)
    qc1.metric("Rows before cleaning",f"{len(raw):,}")
    qc2.metric("Exact duplicate rows removed",f"{duplicate_count:,}")
    qc3.metric("Rows after cleaning",f"{len(df):,}")
    quality = pd.DataFrame({"Column":df.columns,
        "Missing values":df.isna().sum().values,
        "Missing %":(df.isna().mean()*100).round(2).values,
        "Unique values":df.nunique(dropna=True).values})
    st.dataframe(quality, use_container_width=True, hide_index=True)
    st.download_button("Download cleaned filtered data", d.to_csv(index=False).encode("utf-8"),
                       file_name="amarionne_cleaned_filtered_data.csv", mime="text/csv")

st.sidebar.divider()
st.sidebar.caption("Amarionne International · Academic feasibility prototype")
