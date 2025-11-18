# app_compare_final.py
"""
Bank Statement Compare (two-PDF) - Streamlit app
- Upload LEFT and RIGHT PDF statements
- App parses transactions heuristically and produces comparison charts:
    * Monthly spending (lines for each file)
    * Income vs Expense (grouped monthly bars)
    * Category comparison (grouped bars)
    * Top payees side-by-side
- Uses pandas 2.x compatible code (no .iteritems())
- Requirements: streamlit, PyPDF2, pandas, matplotlib
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import PyPDF2
import re
from datetime import datetime
import math

# ---------- Page config ----------
st.set_page_config(layout="wide", page_title="Bank Statement Compare")
plt.rcParams.update({'figure.autolayout': True})

# ---------- User settings (adjust as needed) ----------
FIG_W = 10
FIG_H = 5

# ---------- Category keywords (edit to tune) ----------
CATEGORY_KEYWORDS = {
    "Food & Delivery": ["zomato", "swiggy", "blinkit", "food", "restaurant", "amazon pay"],
    "Subscriptions": ["youtube", "google", "apple", "netflix", "spotify", "subscription", "jio", "airtel"],
    "Utilities": ["tangedco", "electric", "water", "gas", "jio fiber", "airtel postpaid"],
    "Shopping": ["amazon", "flipkart", "neofinity", "mall", "store"],
    "Healthcare": ["clinic", "hospital", "dental", "mediplus"],
    "Cash Withdrawal": ["atm", "cash withdrawal", "cwdr"],
    "Transfers": ["upi", "neft", "rtgs", "imps", "transfer", "paytm", "gpay", "razorpay"]
}

# ---------- Helpers: PDF extraction & parsing ----------
DATE_PATTERNS = [r"\b\d{2}/\d{2}/\d{2,4}\b", r"\b\d{2}-\d{2}-\d{2,4}\b"]

def extract_text_from_pdf(path):
    """Extract text from a PDF using PyPDF2 (best-effort)."""
    text = ""
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for p in reader.pages:
            try:
                txt = p.extract_text() or ""
            except Exception:
                txt = ""
            text += txt + "\n"
    return text

def detect_date_token(s):
    """Return datetime if a date token exists in string s (dayfirst)."""
    for pat in DATE_PATTERNS:
        m = re.search(pat, s)
        if m:
            raw = m.group(0)
            for fmt in ("%d/%m/%y","%d/%m/%Y","%d-%m-%y","%d-%m-%Y"):
                try:
                    return datetime.strptime(raw, fmt)
                except:
                    pass
    return None

def normalize_num_str(s):
    if s is None:
        return None
    s = str(s).replace(",", "").strip()
    try:
        return float(s)
    except:
        return None

def categorize_by_keywords(desc):
    d = (desc or "").lower()
    for cat, keys in CATEGORY_KEYWORDS.items():
        for k in keys:
            if k in d:
                return cat
    return "Other"

def parse_statement_text(text):
    """
    Heuristic parser:
    - Merge continuation lines that lack dates
    - For each line with a date, extract monetary tokens (commas allowed)
    - Infer debit/credit/balance by token positions and (when available) balance deltas
    Returns a DataFrame with Date, Description, Debit, Credit, Balance, Type, Amount, Category.
    """
    raw_lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # merge continuation lines
    lines = []
    for ln in raw_lines:
        if detect_date_token(ln):
            lines.append(ln)
        else:
            if lines:
                lines[-1] = lines[-1] + " " + ln
            else:
                lines.append(ln)

    rows = []
    prev_balance = None
    for ln in lines:
        date = detect_date_token(ln)
        if not date:
            continue
        money_tokens = re.findall(r"-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+\.\d+", ln)
        money_vals = [normalize_num_str(x) for x in money_tokens]
        debit = credit = balance = None
        if len(money_vals) >= 3:
            # common layout: ... withdrawal credit balance
            debit, credit, balance = money_vals[-3], money_vals[-2], money_vals[-1]
        elif len(money_vals) == 2:
            debit, balance = money_vals[0], money_vals[1]
        elif len(money_vals) == 1:
            val = money_vals[0]
            if prev_balance is not None:
                # heuristic relative to previous balance
                if val <= prev_balance * 1.05:
                    debit = val
                else:
                    credit = val
            else:
                debit = val
        # description: strip date and numbers
        desc = re.sub(r"\b\d{2}[\/\-]\d{2}[\/\-]\d{2,4}\b", "", ln)
        desc = re.sub(r"-?\d{1,3}(?:,\d{3})*(?:\.\d+)?", "", desc)
        desc = re.sub(r"\s{2,}", " ", desc).strip()

        rows.append({
            "Date": date.date(),
            "RawLine": ln,
            "Description": desc,
            "Debit": debit,
            "Credit": credit,
            "Balance": balance
        })
        if balance is not None:
            prev_balance = balance

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    for c in ["Debit","Credit","Balance"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Date"] = pd.to_datetime(df["Date"])
    # attempt inference using balance deltas if enough balances exist
    if df["Balance"].notna().sum() >= math.ceil(len(df) * 0.25):
        df = infer_types_from_balances(df)
    else:
        # fallback: set Type by presence
        df["Type"] = df.apply(lambda r: ("Debit" if pd.notna(r["Debit"]) and r["Debit"]>0 else ("Credit" if pd.notna(r["Credit"]) and r["Credit"]>0 else "Other")), axis=1)
        df["Amount"] = df.apply(lambda r: (r["Debit"] if r["Type"]=="Debit" else (r["Credit"] if r["Type"]=="Credit" else 0.0)), axis=1)
    df["Category"] = df["Description"].apply(categorize_by_keywords)
    return df

def infer_types_from_balances(df):
    """Use balance movement to deduce whether a transaction is debit or credit."""
    df = df.copy().reset_index(drop=True)
    df["InferredDebit"] = pd.NA
    df["InferredCredit"] = pd.NA
    for i in range(1, len(df)):
        prev_bal = df.loc[i-1, "Balance"]
        cur_bal = df.loc[i, "Balance"]
        if pd.notna(prev_bal) and pd.notna(cur_bal):
            delta = round(prev_bal - cur_bal, 2)
            if abs(delta) > 0.005:
                if delta > 0:
                    df.loc[i, "InferredDebit"] = delta
                else:
                    df.loc[i, "InferredCredit"] = round(-delta, 2)
    def pick_amount(row):
        if pd.notna(row.get("InferredDebit")):
            return ("Debit", float(row["InferredDebit"]))
        if pd.notna(row.get("InferredCredit")):
            return ("Credit", float(row["InferredCredit"]))
        if pd.notna(row.get("Debit")):
            return ("Debit", float(row["Debit"]))
        if pd.notna(row.get("Credit")):
            return ("Credit", float(row["Credit"]))
        return ("Other", 0.0)
    picks = df.apply(pick_amount, axis=1)
    df["Type"] = [p[0] for p in picks]
    df["Amount"] = [p[1] for p in picks]
    return df

# ---------- Plot helpers ----------
def plot_monthly_spending(dfs, labels):
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    plotted = False
    for df, lbl in zip(dfs, labels):
        if df is None or df.empty:
            continue
        d = df[df["Type"]=="Debit"].copy()
        if d.empty:
            continue
        d["Month"] = d["Date"].dt.to_period("M")
        monthly = d.groupby("Month")["Amount"].sum().sort_index()
        ax.plot(monthly.index.astype(str), monthly.values, marker="o", label=lbl)
        plotted = True
    if not plotted:
        ax.text(0.5, 0.5, "No debit data", ha="center")
        ax.axis("off")
    else:
        ax.set_title("Monthly Spending (Debits)")
        ax.set_xlabel("Month")
        ax.set_ylabel("Amount")
        ax.legend()
        ax.grid(True)
    return fig

def plot_income_vs_expense_side_by_side(df_left, df_right, lbl_left, lbl_right):
    # Build monthly Type sums for left and right and plot grouped bars
    def monthly_agg(df):
        if df is None or df.empty:
            return pd.DataFrame()
        df2 = df.copy()
        df2["Month"] = df2["Date"].dt.to_period("M")
        return df2.groupby(["Month","Type"])["Amount"].sum().unstack(fill_value=0)
    left_agg = monthly_agg(df_left)
    right_agg = monthly_agg(df_right)
    months = sorted(set(list(left_agg.index.astype(str)) + list(right_agg.index.astype(str))))
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    if not months:
        ax.text(0.5,0.5,"No data", ha="center")
        ax.axis("off")
        return fig
    data = []
    for m in months:
        row = {"Month": m}
        for label, agg in (("L", left_agg), ("R", right_agg)):
            if agg.empty or m not in agg.index.astype(str):
                row[f"{label}_Credit"] = 0.0
                row[f"{label}_Debit"] = 0.0
            else:
                row[f"{label}_Credit"] = agg.loc[m].get("Credit", 0.0) if "Credit" in agg.columns else 0.0
                row[f"{label}_Debit"] = agg.loc[m].get("Debit", 0.0) if "Debit" in agg.columns else 0.0
        data.append(row)
    plot_df = pd.DataFrame(data).set_index("Month")
    x = range(len(plot_df))
    width = 0.18
    ax.bar([p - width*1.5 for p in x], plot_df["L_Credit"], width=width, label=f"{lbl_left} - Credit")
    ax.bar([p - width*0.5 for p in x], plot_df["L_Debit"], width=width, label=f"{lbl_left} - Debit")
    ax.bar([p + width*0.5 for p in x], plot_df["R_Credit"], width=width, label=f"{lbl_right} - Credit")
    ax.bar([p + width*1.5 for p in x], plot_df["R_Debit"], width=width, label=f"{lbl_right} - Debit")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df.index, rotation=45)
    ax.set_title("Income vs Expense Comparison (Monthly)")
    ax.set_ylabel("Amount")
    ax.legend()
    return fig

def plot_category_comparison(df_left, df_right, lbl_left, lbl_right, top_n=8):
    def cat_sums(df):
        if df is None or df.empty:
            return pd.Series(dtype=float)
        s = df[df["Type"]=="Debit"].groupby("Category")["Amount"].sum()
        return s
    a = cat_sums(df_left)
    b = cat_sums(df_right)
    combined = (a.fillna(0) + b.fillna(0)).sort_values(ascending=False)
    cats = list(combined.head(top_n).index)
    if not cats:
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
        ax.text(0.5,0.5,"No category data", ha="center")
        ax.axis("off")
        return fig
    a_vals = [a.get(c, 0.0) for c in cats]
    b_vals = [b.get(c, 0.0) for c in cats]
    x = range(len(cats))
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    width = 0.35
    ax.bar([p - width/2 for p in x], a_vals, width=width, label=lbl_left)
    ax.bar([p + width/2 for p in x], b_vals, width=width, label=lbl_right)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=45, ha="right")
    ax.set_title("Category Comparison (Debits)")
    ax.legend()
    return fig

def top_payees_side_by_side(df_left, df_right, top_n=8):
    def top_payees(df, n):
        if df is None or df.empty:
            return pd.Series(dtype=float)
        s = df[df["Type"]=="Debit"].copy()
        s["Payee"] = s["Description"].str.strip().str.replace(r"\s+", " ", regex=True).str[:60]
        return s.groupby("Payee")["Amount"].sum().sort_values(ascending=False).head(n)
    return top_payees(df_left, top_n), top_payees(df_right, top_n)

# ---------- UI: uploads ----------
st.title("Bank Statement Compare — Upload two PDFs")

col_l, col_r = st.columns(2)
with col_l:
    uploaded_left = st.file_uploader("Upload LEFT statement (PDF)", type=["pdf"], key="left")
    label_left = st.text_input("Label for LEFT file", value="LEFT", key="label_left")
with col_r:
    uploaded_right = st.file_uploader("Upload RIGHT statement (PDF)", type=["pdf"], key="right")
    label_right = st.text_input("Label for RIGHT file", value="RIGHT", key="label_right")

if not uploaded_left and not uploaded_right:
    st.info("Upload at least one PDF (left or right) to begin.")
    st.stop()

def handle_upload(uploaded):
    if uploaded is None:
        return None
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded.read())
        tmp.flush()
        path = tmp.name
    text = extract_text_from_pdf(path)
    df = parse_statement_text(text)
    return df

df_left = handle_upload(uploaded_left)
df_right = handle_upload(uploaded_right)

# ---------- Show transaction tables side-by-side ----------
tcol1, tcol2 = st.columns(2)
with tcol1:
    st.subheader(label_left)
    if df_left is None:
        st.write("No file uploaded.")
    elif df_left.empty:
        st.write("No transactions parsed.")
    else:
        st.write(f"Rows parsed: {len(df_left)}")
        st.dataframe(df_left.head(200))
        st.download_button(f"Download {label_left} CSV", df_left.to_csv(index=False), f"{label_left}_transactions.csv", "text/csv")
with tcol2:
    st.subheader(label_right)
    if df_right is None:
        st.write("No file uploaded.")
    elif df_right.empty:
        st.write("No transactions parsed.")
    else:
        st.write(f"Rows parsed: {len(df_right)}")
        st.dataframe(df_right.head(200))
        st.download_button(f"Download {label_right} CSV", df_right.to_csv(index=False), f"{label_right}_transactions.csv", "text/csv")

st.markdown("---")
# ---------- Visualizations ----------
st.subheader("Monthly Spending (Debits) — Comparison")
fig_m = plot_monthly_spending([df_left, df_right], [label_left, label_right])
st.pyplot(fig_m)

st.subheader("Income vs Expense (Monthly) — Comparison")
fig_ie = plot_income_vs_expense_side_by_side(df_left, df_right, label_left, label_right)
st.pyplot(fig_ie)

st.subheader("Category Comparison (Debits)")
fig_cat = plot_category_comparison(df_left, df_right, label_left, label_right)
st.pyplot(fig_cat)

st.subheader("Top Payees (Debits) — Side by Side")
left_top, right_top = top_payees_side_by_side(df_left, df_right, top_n=8)
col_a, col_b = st.columns(2)
with col_a:
    st.markdown(f"**{label_left} — Top payees**")
    if left_top is None or left_top.empty:
        st.write("No data")
    else:
        figL, axL = plt.subplots(figsize=(FIG_W, FIG_H))
        left_top.sort_values().plot(kind="barh", ax=axL)
        axL.set_xlabel("Amount")
        st.pyplot(figL)
with col_b:
    st.markdown(f"**{label_right} — Top payees**")
    if right_top is None or right_top.empty:
        st.write("No data")
    else:
        figR, axR = plt.subplots(figsize=(FIG_W, FIG_H))
        right_top.sort_values().plot(kind="barh", ax=axR)
        axR.set_xlabel("Amount")
        st.pyplot(figR)

st.markdown("---")
st.markdown("**Notes & Tips**")
st.write("""
- Parsing uses heuristics (date token + numeric token position). Different banks/statement layouts may require parser tuning.
- Edit CATEGORY_KEYWORDS at top to improve category detection.
- If payee strings differ slightly between statements you can add a fuzzy-matching step (RapidFuzz) to merge similar payees across files.
""")