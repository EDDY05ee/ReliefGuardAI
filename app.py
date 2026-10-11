import re
from datetime import datetime

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="ReliefGuard AI",
    page_icon="🛡️",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    .stApp {
        background: #f3f6fb;
        color: #172b43;
        font-family: 'DM Sans', sans-serif;
    }
    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #dbe4ef;
    }
    h1, h2, h3 {
        color: #15314f !important;
        font-family: 'Space Grotesk', sans-serif !important;
    }
    .banner {
        background: linear-gradient(120deg, #12314f, #087f72);
        padding: 26px;
        border-radius: 18px;
        margin-bottom: 20px;
    }
    .banner h1 {
        color: white !important;
        margin: 0;
    }
    .banner p {
        color: #edfafa;
        margin-bottom: 0;
    }
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #dbe4ef;
        border-radius: 12px;
        padding: 12px;
    }
    div.stButton > button {
        border-radius: 9px;
        min-height: 42px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =====================================================
# SAMPLE DATA
# =====================================================

DEMO_REPORTS = [
    {
        "id": "R001",
        "incident_type": "Flood",
        "location": "North Market",
        "description": "Flood water entering homes near the market.",
        "urgency": "High",
        "status": "Needs review",
        "submitted_at": "Demo data",
    },
    {
        "id": "R002",
        "incident_type": "Flood",
        "location": "N. Market",
        "description": "Water rising near houses around the north market.",
        "urgency": "High",
        "status": "Needs review",
        "submitted_at": "Demo data",
    },
    {
        "id": "R003",
        "incident_type": "Medical",
        "location": "Central School",
        "description": "An injured resident needs medical assistance.",
        "urgency": "High",
        "status": "New",
        "submitted_at": "Demo data",
    },
    {
        "id": "R004",
        "incident_type": "Road Block",
        "location": "Bridge Road",
        "description": "A fallen tree is blocking the road.",
        "urgency": "Medium",
        "status": "New",
        "submitted_at": "Demo data",
    },
    {
        "id": "R005",
        "incident_type": "Fire",
        "location": "Industrial Area",
        "description": "Smoke and flames reported near a storage building.",
        "urgency": "High",
        "status": "New",
        "submitted_at": "Demo data",
    },
]


REQUIRED_FIELDS = ["incident_type", "location", "description"]


# =====================================================
# DATA HELPERS
# =====================================================

def normalize_location(value):
    text = str(value).strip().lower()

    replacements = [
        (r"\bnw\.", "northwest"),
        (r"\bne\.", "northeast"),
        (r"\bsw\.", "southwest"),
        (r"\bse\.", "southeast"),
        (r"\bn\.", "north"),
        (r"\bs\.", "south"),
        (r"\be\.", "east"),
        (r"\bw\.", "west"),
    ]

    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text)

    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return " ".join(text.split())


def prepare_reports(frame):
    frame = frame.copy()

    frame.columns = [
        str(column).strip().lower().replace(" ", "_")
        for column in frame.columns
    ]

    aliases = {
        "type": "incident_type",
        "incident": "incident_type",
        "place": "location",
        "details": "description",
        "priority": "urgency",
        "report_id": "id",
    }

    rename_map = {
        old: new
        for old, new in aliases.items()
        if old in frame.columns and new not in frame.columns
    }

    frame = frame.rename(columns=rename_map)

    for field in REQUIRED_FIELDS:
        if field not in frame.columns:
            frame[field] = ""

    if "id" not in frame.columns:
        frame["id"] = [
            f"IMP{i:03d}" for i in range(1, len(frame) + 1)
        ]

    if "urgency" not in frame.columns:
        frame["urgency"] = "Unspecified"

    if "status" not in frame.columns:
        frame["status"] = "New"

    if "submitted_at" not in frame.columns:
        frame["submitted_at"] = "Imported"

    for field in [
        "id", "incident_type", "location", "description",
        "urgency", "status", "submitted_at"
    ]:
        frame[field] = frame[field].fillna("").astype(str).str.strip()

    frame["missing_fields"] = frame.apply(
        lambda row: ", ".join(
            field for field in REQUIRED_FIELDS
            if not str(row[field]).strip()
        ),
        axis=1,
    )

    frame["location_normalized"] = frame["location"].apply(
        normalize_location
    )

    return frame


def find_candidates(frame, threshold):
    reports = frame.reset_index(drop=True).copy()

    if len(reports) < 2:
        return pd.DataFrame()

    texts = (
        reports["incident_type"].fillna("").astype(str)
        + " "
        + reports["location_normalized"].fillna("").astype(str)
        + " "
        + reports["description"].fillna("").astype(str)
    ).tolist()

    try:
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
        )
        matrix = vectorizer.fit_transform(texts)
        similarities = cosine_similarity(matrix)
    except ValueError:
        return pd.DataFrame()

    matches = []

    for i in range(len(reports)):
        for j in range(i + 1, len(reports)):
            first = reports.iloc[i]
            second = reports.iloc[j]

            same_type = (
                first["incident_type"].lower()
                == second["incident_type"].lower()
            )

            same_location = (
                bool(first["location_normalized"])
                and first["location_normalized"]
                == second["location_normalized"]
            )

            score = float(similarities[i, j])

            if score >= threshold and (same_type or same_location):
                matches.append({
                    "report_a": first["id"],
                    "report_b": second["id"],
                    "location_a": first["location"],
                    "location_b": second["location"],
                    "similarity": round(score, 3),
                    "description_a": first["description"],
                    "description_b": second["description"],
                })

    return pd.DataFrame(matches)


# =====================================================
# SESSION STATE
# =====================================================

if "reports" not in st.session_state:
    st.session_state["reports"] = prepare_reports(
        pd.DataFrame(DEMO_REPORTS)
    )

if "decisions" not in st.session_state:
    st.session_state["decisions"] = {}

if "threshold" not in st.session_state:
    st.session_state["threshold"] = 0.42

if "submission_count" not in st.session_state:
    st.session_state["submission_count"] = 0


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:
    st.markdown("## 🛡️ ReliefGuard AI")
    st.caption("Incident intelligence • Human-led review")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Report an Incident",
            "Reports & Triage",
            "Human Review",
            "Benchmark Lab",
            "Data & Exports",
        ],
    )

    st.divider()

    st.session_state["threshold"] = st.slider(
        "Similarity threshold",
        min_value=0.10,
        max_value=0.95,
        value=float(st.session_state["threshold"]),
        step=0.05,
    )

    st.caption("Prototype • Demo data • No emergency-service connection")


# =====================================================
# HEADER AND METRICS
# =====================================================

reports = st.session_state["reports"]

candidates = find_candidates(
    reports,
    float(st.session_state["threshold"]),
)

st.markdown(
    """
    <div class="banner">
        <h1>ReliefGuard AI</h1>
        <p>Incident reporting, triage support and human-led review.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

m1, m2, m3, m4 = st.columns(4)

m1.metric("Total reports", len(reports))
m2.metric(
    "Incomplete reports",
    int((reports["missing_fields"] != "").sum()),
)
m3.metric("Candidate pairs", len(candidates))
m4.metric("Human decisions", len(st.session_state["decisions"]))

st.warning(
    "DEMO ONLY: This app does not contact emergency services or "
    "verify incidents. Similarity scores are not probabilities. "
    "For an actual emergency, contact your local emergency services."
)


# =====================================================
# REPORT AN INCIDENT — USER SUBMISSION FORM
# =====================================================

if page == "Report an Incident":
    st.subheader("📝 Report an Incident")

    st.write(
        "Enter the details below. Required fields must be completed "
        "before the report can be submitted."
    )

    with st.form("incident_report_form", clear_on_submit=True):

        incident_type = st.selectbox(
            "Incident type *",
            [
                "Flood",
                "Fire",
                "Medical",
                "Road Block",
                "Landslide",
                "Power Outage",
                "Other",
            ],
        )

        location = st.text_input(
            "Incident location *",
            placeholder="Example: North Market, Main Road",
        )

        description = st.text_area(
            "Describe the incident *",
            placeholder=(
                "What happened? Include useful details "
                "without sharing unnecessary personal information."
            ),
            height=140,
        )

        urgency = st.selectbox(
            "Reported urgency",
            ["High", "Medium", "Low", "Unknown"],
        )

        consent = st.checkbox(
            "I understand this is a demo reporting form, "
            "not an emergency dispatch service."
        )

        submitted = st.form_submit_button(
            "Submit Incident Report",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        missing = []

        if not location.strip():
            missing.append("location")

        if not description.strip():
            missing.append("description")

        if not consent:
            st.error(
                "Please confirm that you understand this is a demo form."
            )

        if missing:
            st.error(
                "Please complete the required field(s): "
                + ", ".join(missing)
                + "."
            )

        elif consent:
            # Generate a unique ID for this session.
            st.session_state["submission_count"] += 1

            report_id = (
                "USR"
                + datetime.now().strftime("%y%m%d%H%M%S")
                + f"{st.session_state['submission_count']:03d}"
            )

            new_report = {
                "id": report_id,
                "incident_type": incident_type,
                "location": location.strip(),
                "description": description.strip(),
                "urgency": urgency,
                "status": "New - awaiting human review",
                "submitted_at": datetime.now().isoformat(
                    timespec="seconds"
                ),
            }

            # Add the submitted report to the current session.
            updated = pd.concat(
                [
                    st.session_state["reports"].drop(
                        columns=["missing_fields", "location_normalized"],
                        errors="ignore",
                    ),
                    pd.DataFrame([new_report]),
                ],
                ignore_index=True,
            )

            st.session_state["reports"] = prepare_reports(updated)

            st.success("Report submitted successfully!")
            st.write("**Your report reference:**", report_id)
            st.info(
                "The report is now in this app session. "
                "It has not been sent to emergency services."
            )

    st.divider()
    st.markdown("### What happens next?")
    st.markdown(
        """
        1. The report is added to the current session.
        2. It appears in Reports & Triage.
        3. The app may surface similar reports for inspection.
        4. A human reviewer makes any final decision.
        """
    )


# =====================================================
# OVERVIEW
# =====================================================

elif page == "Overview":
    st.subheader("Operational Overview")

    left, right = st.columns([1.3, 1])

    with left:
        st.markdown("### Recent reports")

        st.dataframe(
            reports[
                [
                    "id", "incident_type", "location",
                    "urgency", "status",
                ]
            ].tail(10),
            use_container_width=True,
            hide_index=True,
        )

    with right:
        st.markdown("### Workflow")
        st.markdown(
            """
            **01 — Report**  
            Submit an incident through the form.

            **02 — Triage**  
            Inspect reports and potential matches.

            **03 — Human review**  
            Record a decision for candidate pairs.

            **04 — Export**  
            Download reports and review decisions.
            """
        )

    st.info(
        "Sample records are synthetic. Reports submitted in this prototype "
        "are held in session state and are not permanently stored."
    )


# =====================================================
# REPORTS AND TRIAGE
# =====================================================

elif page == "Reports & Triage":
    st.subheader("Reports & Triage")

    col1, col2 = st.columns(2)

    types = ["All"] + sorted(
        [value for value in reports["incident_type"].unique() if value]
    )

    with col1:
        selected_type = st.selectbox("Filter by incident type", types)

    with col2:
        search = st.text_input("Search report ID, location or description")

    view = reports.copy()

    if selected_type != "All":
        view = view[view["incident_type"] == selected_type]

    if search.strip():
        query = search.strip().lower()

        mask = (
            view["id"].str.lower().str.contains(query, regex=False)
            | view["location"].str.lower().str.contains(query, regex=False)
            | view["description"].str.lower().str.contains(query, regex=False)
        )

        view = view[mask]

    st.dataframe(
        view[
            [
                "id", "incident_type", "location", "description",
                "urgency", "status", "missing_fields",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Potentially similar reports")

    if candidates.empty:
        st.info(
            "No candidate pairs found at the current threshold. "
            "This does not prove that reports are unrelated."
        )
    else:
        st.dataframe(
            candidates,
            use_container_width=True,
            hide_index=True,
        )


# =====================================================
# HUMAN REVIEW
# =====================================================

elif page == "Human Review":
    st.subheader("Human Review Queue")

    st.write(
        "Review both reports before recording a decision. "
        "The system never merges or deletes reports automatically."
    )

    if candidates.empty:
        st.info(
            "No candidate pairs are currently available. "
            "Try lowering the threshold in the sidebar."
        )

    else:
        pair_labels = [
            f"{row['report_a']} ↔ {row['report_b']} "
            f"({row['similarity']:.3f})"
            for _, row in candidates.iterrows()
        ]

        selected = st.selectbox("Choose a candidate pair", pair_labels)
        pair = candidates.iloc[pair_labels.index(selected)]

        left, right = st.columns(2)

        with left:
            st.markdown(f"### {pair['report_a']}")
            st.write("**Location:**", pair["location_a"])
            st.write("**Description:**", pair["description_a"])

        with right:
            st.markdown(f"### {pair['report_b']}")
            st.write("**Location:**", pair["location_b"])
            st.write("**Description:**", pair["description_b"])

        st.metric("Text similarity score", f"{pair['similarity']:.3f}")

        st.caption(
            "This score is not the probability that two reports describe "
            "the same event."
        )

        decision_key = f"{pair['report_a']}|{pair['report_b']}"

        options = [
            "Needs review",
            "Likely same incident",
            "Different incidents",
            "Insufficient information",
        ]

        previous = st.session_state["decisions"].get(decision_key, {})

        decision = st.selectbox(
            "Your review decision",
            options,
            index=options.index(previous.get("decision", options[0])),
        )

        note = st.text_area(
            "Review note (optional)",
            value=previous.get("note", ""),
        )

        if st.button("Save review decision", type="primary"):
            st.session_state["decisions"][decision_key] = {
                "report_a": pair["report_a"],
                "report_b": pair["report_b"],
                "similarity": float(pair["similarity"]),
                "decision": decision,
                "note": note.strip(),
                "reviewed_at": datetime.now().isoformat(
                    timespec="seconds"
                ),
            }

            st.success("Review decision saved in this session.")

        if st.session_state["decisions"]:
            st.markdown("### Saved decisions")

            st.dataframe(
                pd.DataFrame(
                    list(st.session_state["decisions"].values())
                ),
                use_container_width=True,
                hide_index=True,
            )


# =====================================================
# BENCHMARK LAB
# =====================================================

elif page == "Benchmark Lab":
    st.subheader("Benchmark Lab")

    st.write(
        "Compare how different thresholds affect the number of "
        "candidate pairs in the current dataset."
    )

    threshold_values = [
        0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80
    ]

    rows = []

    for threshold in threshold_values:
        found = find_candidates(reports, threshold)

        rows.append({
            "threshold": threshold,
            "candidate_pairs": len(found),
        })

    benchmark = pd.DataFrame(rows)

    st.line_chart(
        benchmark.set_index("threshold"),
        y="candidate_pairs",
    )

    st.dataframe(
        benchmark,
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "These results show candidate counts only. They do not establish "
        "precision, recall, or real-world accuracy. Those require "
        "independently labelled test data."
    )


# =====================================================
# DATA AND EXPORTS
# =====================================================

elif page == "Data & Exports":
    st.subheader("Data & Exports")

    uploaded = st.file_uploader(
        "Import incident reports from CSV",
        type=["csv"],
    )

    if uploaded is not None:
        try:
            incoming = pd.read_csv(uploaded)
            prepared = prepare_reports(incoming)

            st.markdown("### Import preview")

            st.dataframe(
                prepared.drop(
                    columns=["location_normalized"],
                    errors="ignore",
                ),
                use_container_width=True,
                hide_index=True,
            )

            if st.button("Replace current reports with imported CSV"):
                st.session_state["reports"] = prepared
                st.session_state["decisions"] = {}
                st.success("Reports replaced for this session.")
                st.rerun()

        except Exception as error:
            st.error(f"Could not import this CSV: {error}")

    export_reports = reports.drop(
        columns=["location_normalized"],
        errors="ignore",
    )

    st.markdown("### Reports")

    st.download_button(
        "Download reports CSV",
        data=export_reports.to_csv(index=False).encode("utf-8"),
        file_name="reliefguard_reports.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.markdown("### Candidate pairs")

    st.download_button(
        "Download candidate pairs CSV",
        data=candidates.to_csv(index=False).encode("utf-8"),
        file_name="reliefguard_candidates.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.markdown("### Human review decisions")

    decision_frame = pd.DataFrame(
        list(st.session_state["decisions"].values())
    )

    st.download_button(
        "Download review decisions CSV",
        data=decision_frame.to_csv(index=False).encode("utf-8"),
        file_name="reliefguard_reviews.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.caption(
        "This prototype uses session state. It does not provide "
        "permanent storage or shared multi-user reporting."
    )


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    "ReliefGuard AI • Hackathon prototype • Human review required • "
    "No live emergency-service connection"
)