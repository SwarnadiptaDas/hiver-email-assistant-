#!/usr/bin/env python3
"""
Hiver AI Email Response System — Interactive Streamlit Dashboard

A beautiful, interactive web dashboard for:
- Generating AI-powered email responses in real-time
- Running multi-dimensional quality evaluations
- Visualizing results with rich charts and tables
- Exploring the dataset and metrics

Usage:
    streamlit run dashboard.py
"""

import sys
import json
import time
import os
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

load_dotenv()

# ─────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Hiver AI Email Assistant",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS for premium look
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f3460 0%, #16213e 100%);
        color: white;
    }
    
    /* Force all text in the sidebar to be white */
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] span, 
    [data-testid="stSidebar"] div, 
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: white !important;
    }
    
    /* Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        backdrop-filter: blur(10px);
        margin-bottom: 1rem;
    }
    
    /* Score badge */
    .score-badge {
        display: inline-block;
        padding: 0.3rem 0.9rem;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1rem;
    }
    .score-high { background: rgba(0,255,136,0.15); color: #00ff88; border: 1px solid #00ff88; }
    .score-mid  { background: rgba(255,200,0,0.15);  color: #ffc800; border: 1px solid #ffc800; }
    .score-low  { background: rgba(255,70,70,0.15);  color: #ff4646; border: 1px solid #ff4646; }

    /* Headings */
    h1, h2, h3 { color: #e0e0ff !important; }
    
    /* General Text */
    p, span, label, .stMarkdown p { color: white !important; }

    /* Input Fields */
    input, textarea, [data-baseweb="input"] input, [data-baseweb="textarea"] textarea {
        color: black !important;
        background-color: white !important;
        border-radius: 6px;
    }
    
    /* Code blocks */
    .email-box {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 10px;
        padding: 1rem 1.2rem;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
        white-space: pre-wrap;
        color: #c0d8ff;
        margin: 0.5rem 0;
    }

    /* Button override */
    .stButton > button {
        background: linear-gradient(135deg, #6c63ff, #3ecfcf);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.6rem 1.4rem;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(108, 99, 255, 0.4);
    }
    
    /* Divider */
    hr { border-color: rgba(255,255,255,0.1); }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Cached Resource Loading
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading retrieval index...")
def load_retriever():
    from generator.retriever import EmailRetriever
    project_root = Path(__file__).resolve().parent
    dataset_path = str(project_root / "dataset" / "email_dataset.json")
    retriever = EmailRetriever(dataset_path=dataset_path)
    retriever.build_index()
    return retriever

@st.cache_resource(show_spinner="Initializing Groq responder...")
def load_responder(_retriever):
    from generator.responder import EmailResponder
    return EmailResponder(retriever=_retriever)

@st.cache_resource(show_spinner="Loading evaluation engine...")
def load_evaluator():
    from evaluation.evaluator import ResponseEvaluator
    return ResponseEvaluator()

@st.cache_data(show_spinner="Loading dataset...")
def load_dataset():
    project_root = Path(__file__).resolve().parent
    dataset_path = project_root / "dataset" / "email_dataset.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────
def score_color(score: float) -> str:
    if score >= 0.75: return "score-high"
    if score >= 0.5:  return "score-mid"
    return "score-low"

def score_emoji(score: float) -> str:
    if score >= 0.75: return "🟢"
    if score >= 0.5:  return "🟡"
    return "🔴"

def render_gauge(score: float, title: str):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score * 100,
        title={"text": title, "font": {"color": "#c0d8ff", "size": 13}},
        number={"suffix": "%", "font": {"color": "#e0e0ff", "size": 22}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#444"},
            "bar": {"color": "#6c63ff"},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 50],   "color": "rgba(255,70,70,0.15)"},
                {"range": [50, 75],  "color": "rgba(255,200,0,0.15)"},
                {"range": [75, 100], "color": "rgba(0,255,136,0.15)"},
            ],
            "threshold": {"line": {"color": "#3ecfcf", "width": 3}, "value": score * 100}
        }
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=170, margin=dict(t=50, b=10, l=10, r=10)
    )
    return fig


# ─────────────────────────────────────────────
# Sidebar Navigation
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📧 Hiver AI")
    st.markdown("**Email Response Assistant**")
    st.markdown("*Powered by Groq + Llama 3.3*")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠  Live Generator", "📊  Evaluate & Score", "📁  Dataset Explorer", "🔬  System Info"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    api_key = os.environ.get("GROQ_API_KEY", "")
    if api_key:
        st.success("Groq API key loaded ✓")
    else:
        st.error("GROQ_API_KEY not set!")
    st.markdown("---")
    st.caption("Hiver Open Challenge · 2024")


# ─────────────────────────────────────────────
# PAGE 1: Live Generator
# ─────────────────────────────────────────────
if "🏠" in page:
    st.title("📧 Live Email Response Generator")
    st.markdown("*Enter a customer support email and get an AI-suggested reply in real time.*")
    st.divider()

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("### ✉️ Incoming Email")
        subject = st.text_input("Subject", placeholder="e.g. Invoice discrepancy for July", key="subject_input")
        body = st.text_area(
            "Email Body",
            height=250,
            placeholder="Paste or type the customer's email here...",
            key="body_input"
        )

        # Quick examples
        with st.expander("📌 Load an example email"):
            examples = {
                "Billing Issue": ("Invoice discrepancy for July",
                    "Hi, I noticed our July invoice shows $2,400 but we're on the Growth plan at $1,800/month. "
                    "Could you explain the difference? Is there an overage fee? We need this resolved before "
                    "accounting closes the month.\n\nThanks, Sarah"),
                "Technical Problem": ("Integration keeps failing",
                    "Hey, I've been trying to set up the Slack integration for 2 hours and keep getting "
                    "'Authentication Failed'. My API credentials look correct. We need this working before "
                    "our standup tomorrow morning. Can someone help urgently?"),
                "Feature Request": ("Can we get bulk export?",
                    "Hi team, our operations team is growing and we'd love a bulk CSV export feature "
                    "for all assigned conversations. Currently we export one at a time which is very slow. "
                    "Is this on your roadmap?"),
            }
            ex_choice = st.selectbox("Choose example", list(examples.keys()))
            if st.button("Load Example"):
                st.session_state["subject_input"] = examples[ex_choice][0]
                st.session_state["body_input"] = examples[ex_choice][1]
                st.rerun()

        generate_btn = st.button("⚡ Generate Response", use_container_width=True)

    with col2:
        st.markdown("### 💬 AI-Suggested Reply")
        result_placeholder = st.empty()
        score_placeholder  = st.empty()

        if generate_btn:
            if not subject.strip() and not body.strip():
                st.warning("Please enter an email subject or body first.")
            else:
                with st.spinner("Retrieving similar examples and generating reply..."):
                    try:
                        retriever = load_retriever()
                        responder = load_responder(retriever)

                        email_obj = {"subject": subject, "body": body, "id": "live_input"}
                        t0 = time.time()
                        response = responder.generate_response(email_obj)
                        elapsed = time.time() - t0

                        reply_body = response.get("body", "")
                        reply_sub  = response.get("subject", "")

                        result_placeholder.markdown(f"""
<div class="metric-card">
<b style="color:#6c63ff;">Subject:</b> <span style="color:#c0d8ff;">{reply_sub}</span><br><br>
<div class="email-box">{reply_body}</div>
<br>
<span style="color:#888; font-size:0.8rem;">⏱️ Generated in {elapsed:.2f}s using Groq Llama-3.3-70b</span>
</div>
""", unsafe_allow_html=True)

                        # Quick evaluation
                        with st.spinner("Scoring quality..."):
                            evaluator = load_evaluator()
                            ev = evaluator.evaluate_single(
                                incoming_email=email_obj,
                                generated_response=reply_body,
                                reference_response=""
                            )
                            cs = ev.get("composite_score", 0.0)
                            score_placeholder.markdown(f"""
<div class="metric-card" style="margin-top:1rem;">
<b style="color:#aaa;">Composite Quality Score</b><br>
<span class="score-badge {score_color(cs)}" style="font-size:1.4rem; margin-top:0.3rem;">
  {score_emoji(cs)} {cs:.0%}
</span>
<span style="color:#888; font-size:0.78rem; margin-left:1rem;">(7-metric weighted evaluation)</span>
</div>
""", unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error: {e}")


# ─────────────────────────────────────────────
# PAGE 2: Evaluate & Score
# ─────────────────────────────────────────────
elif "📊" in page:
    st.title("📊 Batch Evaluation Dashboard")
    st.markdown("*Run the full 7-metric evaluation pipeline on the dataset and visualise results.*")
    st.divider()

    results_path = Path(__file__).resolve().parent / "results" / "evaluation_report.json"

    if results_path.exists():
        with open(results_path, "r", encoding="utf-8") as f:
            report = json.load(f)

        individual = report.get("individual_results", [])
        total      = report.get("total_evaluated", 0)
        overall    = report.get("overall_composite", {})
        per_metric = report.get("per_metric_stats", {})
        per_cat    = report.get("per_category_stats", {})

        # KPI row
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Emails Evaluated",  total)
        kpi2.metric("Mean Composite",    f"{overall.get('mean', 0):.0%}")
        kpi3.metric("Median Score",      f"{overall.get('median', 0):.0%}")
        kpi4.metric("Std Deviation",     f"{overall.get('std', 0):.3f}")
        st.divider()

        # Metric gauges
        st.markdown("### Per-Metric Averages")
        METRIC_LABELS = {
            "semantic_similarity": "Semantic Similarity",
            "intent_coverage":     "Intent Coverage",
            "tone":                "Tone",
            "completeness":        "Completeness",
            "actionability":       "Actionability",
            "fluency":             "Fluency",
            "overall_judge":       "LLM Judge",
        }
        cols = st.columns(len(METRIC_LABELS))
        for i, (key, label) in enumerate(METRIC_LABELS.items()):
            val = per_metric.get(key, {}).get("mean", 0.0)
            cols[i].plotly_chart(render_gauge(val, label), use_container_width=True)

        st.divider()

        # Category breakdown
        if per_cat:
            st.markdown("### Category Breakdown")
            cat_df = pd.DataFrame([
                {"Category": k.capitalize(), "Mean Score": v["mean"], "Count": v["count"]}
                for k, v in per_cat.items()
            ]).sort_values("Mean Score", ascending=False)

            fig_cat = px.bar(
                cat_df, x="Category", y="Mean Score", color="Mean Score",
                color_continuous_scale=["#ff4646", "#ffc800", "#00ff88"],
                range_color=[0, 1], text="Mean Score",
                title="Mean Composite Score by Category"
            )
            fig_cat.update_traces(texttemplate="%{text:.0%}", textposition="outside")
            fig_cat.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#c0d8ff", title_font_color="#e0e0ff",
                yaxis=dict(range=[0, 1.15], tickformat=".0%"),
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_cat, use_container_width=True)

        st.divider()

        # Per-response table
        if individual:
            st.markdown("### Per-Response Score Table")
            rows = []
            for ev in individual:
                metrics = ev.get("metrics", {})
                def g(k): return metrics.get(k, {}).get("score", 0.0) if isinstance(metrics.get(k), dict) else 0.0
                rows.append({
                    "Email ID":  ev.get("email_id", "?"),
                    "Category":  ev.get("category", "?").capitalize(),
                    "Semantic":  g("semantic_similarity"),
                    "Intent":    g("intent_coverage"),
                    "Tone":      g("tone"),
                    "Complete":  g("completeness"),
                    "Action":    g("actionability"),
                    "Fluency":   g("fluency"),
                    "Judge":     g("overall_judge"),
                    "COMPOSITE": ev.get("composite_score", 0.0),
                })
            df = pd.DataFrame(rows)

            def color_val(v):
                if isinstance(v, float):
                    if v >= 0.75: return "color: #00ff88"
                    if v >= 0.5:  return "color: #ffc800"
                    return "color: #ff4646"
                return ""

            styled = df.style.applymap(
                color_val, subset=["Semantic", "Intent", "Tone", "Complete", "Action", "Fluency", "Judge", "COMPOSITE"]
            ).format({
                "Semantic": "{:.0%}", "Intent": "{:.0%}", "Tone": "{:.0%}",
                "Complete": "{:.0%}", "Action": "{:.0%}", "Fluency": "{:.0%}",
                "Judge": "{:.0%}", "COMPOSITE": "{:.0%}",
            })
            st.dataframe(styled, use_container_width=True, height=300)

    else:
        st.info("No evaluation report found yet. Run `python evaluate.py` first to generate results, then refresh.")


# ─────────────────────────────────────────────
# PAGE 3: Dataset Explorer
# ─────────────────────────────────────────────
elif "📁" in page:
    st.title("📁 Dataset Explorer")
    st.markdown("*Browse the 50-email curated support dataset.*")
    st.divider()

    data = load_dataset()
    emails = data.get("emails", [])
    meta   = data.get("metadata", {})

    # Summary
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Email Pairs", meta.get("total_count", len(emails)))
    m2.metric("Categories",        len(meta.get("categories", {})))
    m3.metric("Generated At",      meta.get("generated_at", "N/A")[:10])

    st.divider()

    # Category distribution chart
    cats = meta.get("categories", {})
    if cats:
        cat_df = pd.DataFrame({"Category": list(cats.keys()), "Count": list(cats.values())})
        fig_pie = px.pie(
            cat_df, names="Category", values="Count",
            title="Category Distribution",
            color_discrete_sequence=px.colors.sequential.Plasma_r,
            hole=0.4
        )
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", font_color="#c0d8ff",
            title_font_color="#e0e0ff", legend_font_color="#c0d8ff"
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.divider()

    # Browseable table
    st.markdown("### Browse Email Pairs")
    categories = sorted(set(e.get("category", "") for e in emails))
    selected_cat = st.selectbox("Filter by category", ["All"] + categories)

    filtered = emails if selected_cat == "All" else [e for e in emails if e.get("category") == selected_cat]

    for i, email in enumerate(filtered[:20]):  # limit to 20 for performance
        inc = email.get("incoming_email", {})
        ref = email.get("reference_response", {})
        with st.expander(f"📩 {email.get('id', '?')} — {inc.get('subject', 'No subject')}"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Incoming Email**")
                st.markdown(f"*From:* {inc.get('sender_email', '?')}")
                st.markdown(f"*Subject:* {inc.get('subject', '?')}")
                st.markdown(f"""<div class="email-box">{inc.get('body', '')}</div>""", unsafe_allow_html=True)
            with c2:
                st.markdown("**Reference Response**")
                st.markdown(f"""<div class="email-box">{ref.get('body', '')}</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE 4: System Info
# ─────────────────────────────────────────────
elif "🔬" in page:
    st.title("🔬 System Information")
    st.divider()

    st.markdown("### Architecture")
    st.markdown("""
| Component | Technology | Purpose |
|---|---|---|
| **Embedding Model** | `all-MiniLM-L6-v2` (local) | Encode emails into dense vectors |
| **Vector Store** | FAISS (CPU) | Fast nearest-neighbor retrieval |
| **LLM (Generation)** | Groq — Llama-3.3-70b-versatile | Generate high-quality support replies |
| **LLM (Evaluation)** | Groq — Llama-3.3-70b-versatile | Score 6 of 7 quality dimensions |
| **Semantic Eval** | scikit-learn cosine similarity | Embedding-based similarity metric |
| **Dashboard** | Streamlit + Plotly | Interactive visual interface |
    """)

    st.divider()
    st.markdown("### Evaluation Weights")
    weights = {
        "Intent Coverage": 0.25, "Completeness": 0.20,
        "Semantic Similarity": 0.15, "Tone": 0.15,
        "Actionability": 0.10, "LLM Judge": 0.10, "Fluency": 0.05
    }
    wdf = pd.DataFrame({"Metric": list(weights.keys()), "Weight": list(weights.values())})
    fig_w = px.bar(
        wdf, x="Weight", y="Metric", orientation="h",
        color="Weight", color_continuous_scale=["#3ecfcf", "#6c63ff"],
        title="Evaluation Metric Weights"
    )
    fig_w.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#c0d8ff", title_font_color="#e0e0ff", coloraxis_showscale=False,
        xaxis=dict(tickformat=".0%")
    )
    fig_w.update_traces(texttemplate="%{x:.0%}", textposition="outside")
    st.plotly_chart(fig_w, use_container_width=True)

    st.divider()
    st.markdown("### RAG Pipeline Flow")
    st.markdown("""
```
Incoming Email
      │
      ▼
 [Embedding Model]  ──→  Dense Vector (384-dim)
      │
      ▼
 [FAISS Index]  ──→  Top-3 Similar Past Emails
      │
      ▼
 [Prompt Builder]  ──→  Few-shot Prompt with Context
      │
      ▼
 [Groq Llama-3.3]  ──→  Generated Response Body
      │
      ▼
 [7-Metric Evaluator]  ──→  Composite Quality Score
```
    """)
