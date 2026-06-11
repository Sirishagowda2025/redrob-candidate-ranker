"""
Redrob Candidate Ranker — Streamlit Sandbox
Hackathon sandbox demo: accepts ≤100 candidates JSON, runs the ranker, returns CSV.

Deploy to Streamlit Cloud:
  streamlit run app.py
"""

import csv
import io
import json
import sys
from pathlib import Path

import streamlit as st

# Import the ranker (must be in same directory)
sys.path.insert(0, str(Path(__file__).parent))
from rank import score_candidate, generate_reasoning

st.set_page_config(
    page_title="Redrob Candidate Ranker",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Redrob Intelligent Candidate Ranker")
st.caption("Hackathon sandbox · Ranks candidates for the Senior AI Engineer JD · CPU-only, no APIs")

st.markdown("""
**Architecture:** Three-stage pipeline — (1) Semantic skill match against JD, 
(2) Career fit scoring (YOE, company type, title, production signals), 
(3) Behavioral availability multiplier. Honeypot detection included.
""")

st.divider()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Input")
    uploaded = st.file_uploader(
        "Upload candidates JSON (array of candidates, max 100)",
        type=["json", "jsonl"],
        help="Use sample_candidates.json from the hackathon bundle, or any JSON array of candidates"
    )
    
    use_sample = st.checkbox("Use built-in 5 sample candidates for demo")

SAMPLE_CANDIDATES = [
    {
        "candidate_id": "CAND_DEMO001",
        "profile": {
            "anonymized_name": "Demo Candidate A",
            "headline": "Senior ML Engineer | Embeddings, Vector Search, Ranking",
            "summary": "7 years building recommendation and ranking systems at product companies.",
            "location": "Bengaluru", "country": "India",
            "years_of_experience": 7.0, "current_title": "ML Engineer",
            "current_company": "Swiggy", "current_company_size": "1001-5000",
            "current_industry": "Food Tech"
        },
        "career_history": [
            {"company": "Swiggy", "title": "ML Engineer", "start_date": "2022-01-01",
             "end_date": None, "duration_months": 28, "is_current": True,
             "industry": "Food Tech", "company_size": "1001-5000",
             "description": "Built recommendation ranking system using FAISS and sentence-transformers. Deployed to production serving 10M users. Implemented NDCG evaluation framework for offline testing."},
            {"company": "Flipkart", "title": "Data Scientist", "start_date": "2018-06-01",
             "end_date": "2022-01-01", "duration_months": 43, "is_current": False,
             "industry": "E-Commerce", "company_size": "10001+",
             "description": "Search relevance team. Built hybrid BM25+dense retrieval pipeline. A/B tested ranking changes. Shipped to 100M+ users."}
        ],
        "education": [{"institution": "IIT Bombay", "degree": "B.Tech", "field_of_study": "CS", "start_year": 2014, "end_year": 2018, "tier": "tier_1"}],
        "skills": [
            {"name": "Embeddings", "proficiency": "advanced", "endorsements": 45, "duration_months": 36},
            {"name": "FAISS", "proficiency": "advanced", "endorsements": 30, "duration_months": 28},
            {"name": "Pinecone", "proficiency": "intermediate", "endorsements": 12, "duration_months": 18},
            {"name": "Python", "proficiency": "advanced", "endorsements": 80, "duration_months": 84},
            {"name": "Sentence Transformers", "proficiency": "advanced", "endorsements": 25, "duration_months": 24},
        ],
        "certifications": [], "languages": [{"language": "English", "proficiency": "professional"}],
        "redrob_signals": {
            "profile_completeness_score": 92, "signup_date": "2024-01-01",
            "last_active_date": "2026-06-08", "open_to_work_flag": True,
            "profile_views_received_30d": 45, "applications_submitted_30d": 3,
            "recruiter_response_rate": 0.85, "avg_response_time_hours": 6,
            "skill_assessment_scores": {"Embeddings": 88, "Python": 90},
            "connection_count": 450, "endorsements_received": 192,
            "notice_period_days": 30, "expected_salary_range_inr_lpa": {"min": 35, "max": 55},
            "preferred_work_mode": "hybrid", "willing_to_relocate": True,
            "github_activity_score": 72, "search_appearance_30d": 180,
            "saved_by_recruiters_30d": 8, "interview_completion_rate": 0.92,
            "offer_acceptance_rate": 0.75, "verified_email": True,
            "verified_phone": True, "linkedin_connected": True
        }
    },
    {
        "candidate_id": "CAND_DEMO002",
        "profile": {
            "anonymized_name": "Demo Candidate B — Ghost",
            "headline": "AI Engineer | RAG, LLMs, Vector Search",
            "summary": "Expert in all things AI. RAG, embeddings, fine-tuning, vector databases.",
            "location": "Pune", "country": "India",
            "years_of_experience": 6.5, "current_title": "AI Engineer",
            "current_company": "Startup X", "current_company_size": "51-200",
            "current_industry": "SaaS"
        },
        "career_history": [
            {"company": "Startup X", "title": "AI Engineer", "start_date": "2021-06-01",
             "end_date": None, "duration_months": 36, "is_current": True,
             "industry": "SaaS", "company_size": "51-200",
             "description": "Worked on LLM integration and RAG systems. Used Qdrant and Weaviate."}
        ],
        "education": [{"institution": "BITS Pilani", "degree": "B.E.", "field_of_study": "CS", "start_year": 2016, "end_year": 2020, "tier": "tier_1"}],
        "skills": [
            {"name": "RAG", "proficiency": "advanced", "endorsements": 20, "duration_months": 24},
            {"name": "Qdrant", "proficiency": "advanced", "endorsements": 15, "duration_months": 18},
            {"name": "LLM Fine-tuning", "proficiency": "advanced", "endorsements": 18, "duration_months": 20},
        ],
        "certifications": [], "languages": [{"language": "English", "proficiency": "professional"}],
        "redrob_signals": {
            "profile_completeness_score": 78, "signup_date": "2023-03-01",
            "last_active_date": "2025-11-01",  # 7 months ago = ghost
            "open_to_work_flag": False,
            "profile_views_received_30d": 5, "applications_submitted_30d": 0,
            "recruiter_response_rate": 0.05, "avg_response_time_hours": 240,
            "skill_assessment_scores": {}, "connection_count": 120,
            "endorsements_received": 53, "notice_period_days": 90,
            "expected_salary_range_inr_lpa": {"min": 30, "max": 50},
            "preferred_work_mode": "remote", "willing_to_relocate": False,
            "github_activity_score": -1, "search_appearance_30d": 12,
            "saved_by_recruiters_30d": 1, "interview_completion_rate": 0.4,
            "offer_acceptance_rate": 0.2, "verified_email": True,
            "verified_phone": False, "linkedin_connected": False
        }
    },
    {
        "candidate_id": "CAND_DEMO003",
        "profile": {
            "anonymized_name": "Demo Candidate C — Keyword Stuffer",
            "headline": "Marketing Manager | AI enthusiast",
            "summary": "Marketing professional with deep interest in AI technologies.",
            "location": "Mumbai", "country": "India",
            "years_of_experience": 8.0, "current_title": "Marketing Manager",
            "current_company": "FMCG Corp", "current_company_size": "10001+",
            "current_industry": "Consumer Goods"
        },
        "career_history": [
            {"company": "FMCG Corp", "title": "Marketing Manager", "start_date": "2019-01-01",
             "end_date": None, "duration_months": 78, "is_current": True,
             "industry": "Consumer Goods", "company_size": "10001+",
             "description": "Lead marketing campaigns and brand strategy. Explored AI tools for marketing analytics."}
        ],
        "education": [{"institution": "MDI Gurgaon", "degree": "MBA", "field_of_study": "Marketing", "start_year": 2015, "end_year": 2017, "tier": "tier_2"}],
        "skills": [
            {"name": "RAG", "proficiency": "expert", "endorsements": 55, "duration_months": 12},
            {"name": "FAISS", "proficiency": "expert", "endorsements": 48, "duration_months": 10},
            {"name": "Pinecone", "proficiency": "expert", "endorsements": 42, "duration_months": 8},
            {"name": "PyTorch", "proficiency": "expert", "endorsements": 60, "duration_months": 15},
            {"name": "Weaviate", "proficiency": "advanced", "endorsements": 38, "duration_months": 9},
            {"name": "NDCG", "proficiency": "advanced", "endorsements": 30, "duration_months": 6},
        ],
        "certifications": [], "languages": [{"language": "English", "proficiency": "professional"}],
        "redrob_signals": {
            "profile_completeness_score": 88, "signup_date": "2025-01-01",
            "last_active_date": "2026-06-09", "open_to_work_flag": True,
            "profile_views_received_30d": 25, "applications_submitted_30d": 8,
            "recruiter_response_rate": 0.9, "avg_response_time_hours": 2,
            "skill_assessment_scores": {"RAG": 42, "FAISS": 38},  # Low assessments despite "expert" claim
            "connection_count": 800, "endorsements_received": 273,
            "notice_period_days": 30, "expected_salary_range_inr_lpa": {"min": 25, "max": 45},
            "preferred_work_mode": "hybrid", "willing_to_relocate": True,
            "github_activity_score": 3, "search_appearance_30d": 95,
            "saved_by_recruiters_30d": 3, "interview_completion_rate": 0.75,
            "offer_acceptance_rate": 0.5, "verified_email": True,
            "verified_phone": True, "linkedin_connected": True
        }
    },
    {
        "candidate_id": "CAND_DEMO004",
        "profile": {
            "anonymized_name": "Demo Candidate D — Consulting Only",
            "headline": "Data Scientist | TCS | AI/ML",
            "summary": "8 years at TCS delivering AI solutions for enterprise clients.",
            "location": "Hyderabad", "country": "India",
            "years_of_experience": 8.0, "current_title": "Senior Data Scientist",
            "current_company": "TCS", "current_company_size": "10001+",
            "current_industry": "IT Services"
        },
        "career_history": [
            {"company": "TCS", "title": "Senior Data Scientist", "start_date": "2020-01-01",
             "end_date": None, "duration_months": 66, "is_current": True,
             "industry": "IT Services", "company_size": "10001+",
             "description": "Built ML models for banking client. Used embeddings and NLP for document classification."},
            {"company": "Infosys", "title": "Data Analyst", "start_date": "2017-07-01",
             "end_date": "2019-12-01", "duration_months": 29, "is_current": False,
             "industry": "IT Services", "company_size": "10001+",
             "description": "Analytics and reporting for BFSI clients. SQL, Python, Tableau."}
        ],
        "education": [{"institution": "JNTU", "degree": "B.Tech", "field_of_study": "CS", "start_year": 2013, "end_year": 2017, "tier": "tier_3"}],
        "skills": [
            {"name": "Embeddings", "proficiency": "intermediate", "endorsements": 22, "duration_months": 30},
            {"name": "Python", "proficiency": "advanced", "endorsements": 65, "duration_months": 84},
            {"name": "NLP", "proficiency": "intermediate", "endorsements": 35, "duration_months": 36},
        ],
        "certifications": [], "languages": [{"language": "English", "proficiency": "professional"}],
        "redrob_signals": {
            "profile_completeness_score": 82, "signup_date": "2024-06-01",
            "last_active_date": "2026-05-28", "open_to_work_flag": True,
            "profile_views_received_30d": 30, "applications_submitted_30d": 5,
            "recruiter_response_rate": 0.7, "avg_response_time_hours": 18,
            "skill_assessment_scores": {"Python": 76, "NLP": 58},
            "connection_count": 340, "endorsements_received": 122,
            "notice_period_days": 60, "expected_salary_range_inr_lpa": {"min": 20, "max": 38},
            "preferred_work_mode": "hybrid", "willing_to_relocate": True,
            "github_activity_score": 25, "search_appearance_30d": 140,
            "saved_by_recruiters_30d": 5, "interview_completion_rate": 0.85,
            "offer_acceptance_rate": 0.65, "verified_email": True,
            "verified_phone": True, "linkedin_connected": True
        }
    },
    {
        "candidate_id": "CAND_DEMO005",
        "profile": {
            "anonymized_name": "Demo Candidate E — Good Fit",
            "headline": "Applied Scientist | Search & Ranking | NLP",
            "summary": "6 years building search and recommendation systems at product companies. Expert in hybrid retrieval.",
            "location": "Noida", "country": "India",
            "years_of_experience": 6.2, "current_title": "Applied Scientist",
            "current_company": "Naukri.com", "current_company_size": "1001-5000",
            "current_industry": "HR Tech"
        },
        "career_history": [
            {"company": "Naukri.com", "title": "Applied Scientist", "start_date": "2021-03-01",
             "end_date": None, "duration_months": 39, "is_current": True,
             "industry": "HR Tech", "company_size": "1001-5000",
             "description": "Owned ranking for candidate-JD matching. Built hybrid BM25 + dense retrieval pipeline using Elasticsearch and sentence-transformers. Implemented NDCG evaluation framework and A/B testing infrastructure. Shipped v3 ranking which improved recruiter engagement by 35%."},
            {"company": "Amazon India", "title": "SDE II (ML)", "start_date": "2019-01-01",
             "end_date": "2021-02-01", "duration_months": 25, "is_current": False,
             "industry": "E-Commerce", "company_size": "10001+",
             "description": "Product recommendation team. Learning-to-rank with XGBoost. Deployed to production at scale."}
        ],
        "education": [{"institution": "NIT Trichy", "degree": "B.Tech", "field_of_study": "CS", "start_year": 2015, "end_year": 2019, "tier": "tier_1"}],
        "skills": [
            {"name": "Elasticsearch", "proficiency": "advanced", "endorsements": 55, "duration_months": 36},
            {"name": "Sentence Transformers", "proficiency": "advanced", "endorsements": 38, "duration_months": 30},
            {"name": "BM25", "proficiency": "advanced", "endorsements": 28, "duration_months": 36},
            {"name": "NDCG", "proficiency": "advanced", "endorsements": 20, "duration_months": 36},
            {"name": "XGBoost", "proficiency": "advanced", "endorsements": 32, "duration_months": 25},
            {"name": "Python", "proficiency": "advanced", "endorsements": 90, "duration_months": 74},
        ],
        "certifications": [], "languages": [{"language": "English", "proficiency": "professional"}],
        "redrob_signals": {
            "profile_completeness_score": 95, "signup_date": "2023-12-01",
            "last_active_date": "2026-06-09", "open_to_work_flag": True,
            "profile_views_received_30d": 68, "applications_submitted_30d": 4,
            "recruiter_response_rate": 0.88, "avg_response_time_hours": 5,
            "skill_assessment_scores": {"Elasticsearch": 85, "Python": 92, "NDCG": 88},
            "connection_count": 520, "endorsements_received": 263,
            "notice_period_days": 45, "expected_salary_range_inr_lpa": {"min": 40, "max": 65},
            "preferred_work_mode": "hybrid", "willing_to_relocate": False,
            "github_activity_score": 60, "search_appearance_30d": 220,
            "saved_by_recruiters_30d": 12, "interview_completion_rate": 0.95,
            "offer_acceptance_rate": 0.8, "verified_email": True,
            "verified_phone": True, "linkedin_connected": True
        }
    }
]

with col2:
    st.subheader("📊 Ranked Output")

    candidates = None

    if use_sample:
        candidates = SAMPLE_CANDIDATES
        st.info(f"Using {len(candidates)} built-in demo candidates")
    elif uploaded:
        try:
            content = uploaded.read().decode("utf-8")
            if content.strip().startswith("["):
                candidates = json.loads(content)
            else:
                candidates = [json.loads(line) for line in content.strip().split("\n") if line.strip()]
            if len(candidates) > 100:
                st.warning(f"Truncating to first 100 candidates (uploaded {len(candidates)})")
                candidates = candidates[:100]
            st.info(f"Loaded {len(candidates)} candidates")
        except Exception as e:
            st.error(f"Error parsing file: {e}")

    if candidates:
        # Score all
        scored = []
        for c in candidates:
            score, breakdown = score_candidate(c)
            scored.append((score, breakdown, c))
        scored.sort(key=lambda x: (-x[0], x[2].get("candidate_id", "")))

        # Build output CSV
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])

        rows_display = []
        for rank, (score, breakdown, c) in enumerate(scored, 1):
            cid = c.get("candidate_id", "")
            reasoning = generate_reasoning(c, breakdown, rank)
            writer.writerow([cid, rank, f"{score:.6f}", reasoning])
            p = c.get("profile", {})
            rows_display.append({
                "Rank": rank,
                "ID": cid,
                "Title": p.get("current_title", "?"),
                "YOE": p.get("years_of_experience", "?"),
                "Score": f"{score:.4f}",
                "Availability": f"{breakdown.get('availability_score', 0):.2f}",
                "Honeypot": "⚠️" if breakdown.get("honeypot_penalty", 1) < 0.5 else "✅",
                "Reasoning": reasoning[:120] + "..." if len(reasoning) > 120 else reasoning
            })

        import pandas as pd
        df = pd.DataFrame(rows_display)
        st.dataframe(df, use_container_width=True, hide_index=True)

        csv_str = output.getvalue()
        st.download_button(
            "⬇️ Download Submission CSV",
            data=csv_str.encode("utf-8"),
            file_name="submission.csv",
            mime="text/csv"
        )

        # Score breakdown for top candidate
        if scored:
            st.markdown("---")
            st.subheader(f"🔍 Score Breakdown — Rank #1")
            top_score, top_breakdown, top_c = scored[0]
            p = top_c.get("profile", {})
            c1, c2, c3 = st.columns(3)
            c1.metric("Skill Match", f"{top_breakdown.get('skill_composite', 0):.3f}")
            c2.metric("Career Fit", f"{top_breakdown.get('career_composite', 0):.3f}")
            c3.metric("Behavioral", f"{top_breakdown.get('behavioral_composite', 0):.3f}")
            c1.metric("Availability Mult.", f"{top_breakdown.get('availability_multiplier', 0):.3f}")
            c2.metric("Honeypot Penalty", f"{top_breakdown.get('honeypot_penalty', 1):.2f}×")
            c3.metric("Final Score", f"{top_score:.4f}")
    else:
        st.info("Upload a candidates JSON file or check 'Use built-in sample' to see the ranker in action.")

st.divider()
st.caption("Redrob Hackathon 2026 · Sirisha Gowda · github.com/Sirishagowda2025/redrob-candidate-ranker · CPU-only · No API calls")
