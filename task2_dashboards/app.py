import streamlit as st
import pandas as pd
import google.generativeai as genai
from groq import Groq
import os
from datetime import datetime
from typing import Optional, List, Dict


# Constants & Configuration

PAGE_TITLE = "Feedback AI Ecosystem"
PAGE_ICON = "🤖"
DATA_FILE = os.path.join("task2_dashboards", "data", "feedback.csv")

# Prompts
PROMPT_USER_RESPONSE = (
    "Act as a senior customer success manager. A customer named '{name}' has provided the following feedback: "
    "Rating: {rating}/5. Review: '{review}'. "
    "Write a concise, warm, and professional response to them, addressing them by their name. "
    "Sign off as 'The Customer Success Team'. "
    "If the rating is low, be apologetic and ask for a chance to fix it. "
    "If high, thank them enthusiastically."
)

PROMPT_ADMIN_SUMMARY = (
    "Analyze the following customer review: '{review}'. "
    "Provide a strictly factual summary in exactly one sentence (max 15 words)."
)

PROMPT_ADMIN_ACTION = (
    "Based on the following review: '{review}' (Rating: {rating}/5), "
    "suggest ONE specific, actionable operational improvement for the business. "
    "Start with a verb. Keep it under 10 words."
)

st.set_page_config(page_title=PAGE_TITLE, layout="wide", page_icon=PAGE_ICON)


# Service Layer (Logic & Data)


class FeedbackService:
    """Handles data persistence and AI interactions."""
    
    EXPECTED_COLS = [
        "Timestamp", "Name", "Rating", "Review", "Gender", "Age_Group",
        "User_Response", "Admin_Summary", "Admin_Action"
    ]

    @staticmethod
    def get_api_key() -> Optional[str]:
        """Retrieves API key specifically from Streamlit secrets (Cloud Deployment)."""
        # Per strict deployment instructions:
        return st.secrets["GOOGLE_API_KEY"]

    @staticmethod
    def generate_ai_content(prompt: str, api_key: str) -> str:
        """
        Generates content using Gemini with automatic model fallback.
        Retries across multiple model variants to ensure high availability.
        """
        genai.configure(api_key=api_key)
        # Priority list: Fastest/Cheapest -> Standard -> Legacy
        models_to_try = ['gemini-1.5-flash', 'gemini-pro', 'gemini-1.0-pro']
        
        for model_name in models_to_try:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                return response.text
            except Exception:
                continue 

        # --- Fallback to Groq ---
        try:
            # Fallback to Groq using strict st.secrets as per instructions
            groq_key = st.secrets.get("GROQ_API_KEY", None)
            if groq_key:
                client = Groq(api_key=groq_key)
                completion = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7
                )
                return completion.choices[0].message.content
        except Exception:
            pass
        
        return "⚠️ System Note: AI Service currently unavailable. Please check quotas."

    @classmethod
    def load_data(cls) -> pd.DataFrame:
        """Loads the dataset ensuring schema consistency."""
        if os.path.exists(DATA_FILE):
            try:
                df = pd.read_csv(DATA_FILE)
                # Schema validation/migration
                for col in cls.EXPECTED_COLS:
                    if col not in df.columns:
                        df[col] = "Not Specified" # Default for old records
                return df
            except Exception:
                pass # Fallback to empty
        
        return pd.DataFrame(columns=cls.EXPECTED_COLS)

    @classmethod
    def save_entry(cls, entry: Dict[str, str]) -> None:
        """Appends a new entry to the CSV storage."""
        df = cls.load_data()
        new_row = pd.DataFrame([entry])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(DATA_FILE, index=False)


# UI Layer (Views)


def render_user_dashboard(api_key: str):
    """Renders the public-facing submission interface."""
    st.title(f"{PAGE_ICON} Share Your Experience")
    st.markdown("### We value your feedback")
    st.write("Help us improve by sharing your experience below.")

    with st.container(border=True):
        name = st.text_input("Full Name", placeholder="John Doe")
        
        # Demographics
        c1, c2 = st.columns(2)
        with c1:
            gender = st.selectbox("Gender", ["Select...", "Male", "Female", "Non-binary", "Prefer not to say"])
        with c2:
            age_group = st.selectbox("Age Group", ["Select...", "Under 18", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"])

        rating = st.feedback("stars")
        if rating is None:
            rating = 4 
            star_val = 5
        else:
            star_val = rating + 1

        review = st.text_area("Your comments", placeholder="Please share details about your experience...", height=150)
        
        submit_clicked = st.button("Submit Feedback", type="primary", use_container_width=True)

    if submit_clicked:
        if not review.strip():
            st.warning("Please provide a short text review to proceed.")
            return
            
        if not name.strip():
            st.warning("Please enter your name.")
            return

        with st.status("Processing your feedback...", expanded=True) as status:
            # 1. Generate User Response
            st.write("Generating personalized response...")
            user_resp_prompt = PROMPT_USER_RESPONSE.format(name=name, rating=star_val, review=review)
            ai_response = FeedbackService.generate_ai_content(user_resp_prompt, api_key)
            
            # 2. Generate Admin Metadata (Live)
            st.write("Analyzing for internal quality control...")
            summary_prompt = PROMPT_ADMIN_SUMMARY.format(review=review)
            action_prompt = PROMPT_ADMIN_ACTION.format(review=review, rating=star_val)
            
            admin_summary = FeedbackService.generate_ai_content(summary_prompt, api_key)
            admin_action = FeedbackService.generate_ai_content(action_prompt, api_key)
            
            # 3. Persist
            entry = {
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Name": name,
                "Rating": star_val,
                "Review": review,
                "Gender": gender if gender != "Select..." else "Not Specified",
                "Age_Group": age_group if age_group != "Select..." else "Not Specified",
                "User_Response": ai_response,
                "Admin_Summary": admin_summary,
                "Admin_Action": admin_action
            }
            FeedbackService.save_entry(entry)
            
            status.update(label="Feedback Recorded Successfully!", state="complete", expanded=False)

        st.divider()
        st.subheader("Our Response")
        st.success(ai_response, icon="📨")


def render_admin_dashboard():
    """Renders the internal analytics and data view."""
    st.title("📊 Executive Dashboard")
    st.markdown("Live view of incoming customer sentiment and operational insights.")
    
    df = FeedbackService.load_data()
    
    if df.empty:
        st.info("No data available yet. Waiting for submissions.")
        return

    # --- Analytics Header ---
    with st.container():
        c1, c2, c3, c4 = st.columns(4)
        
        c1.metric("Total Reviews", len(df))
        c2.metric("Avg Rating", f"{df['Rating'].mean():.1f} ⭐")
        
        positive_pct = (len(df[df['Rating'] >= 4]) / len(df)) * 100
        c3.metric("Positive Sentiment", f"{positive_pct:.0f}%")
        
        last_time = pd.to_datetime(df["Timestamp"]).max()
        time_diff = datetime.now() - last_time
        time_str = "Just now" if time_diff.total_seconds() < 60 else f"{int(time_diff.total_seconds()//60)}m ago"
        c4.metric("Last Activity", time_str)

    # --- Charts ---
    st.markdown("### Demographics & Trends")
    c_chart1, c_chart2, c_chart3 = st.columns(3)
    
    with c_chart1:
        st.caption("Rating Distribution")
        chart_data = df['Rating'].value_counts().reindex(range(1, 6), fill_value=0).reset_index()
        chart_data.columns = ['Stars', 'Count']
        st.bar_chart(chart_data.set_index('Stars'), color="#4CAF50")
    
    with c_chart2:
        st.caption("Gender Demographics")
        if "Gender" in df.columns:
            gender_counts = df['Gender'].value_counts()
            st.bar_chart(gender_counts, horizontal=True) # Horizontal bars better for labels
            
    with c_chart3:
        st.caption("Age Group Distribution")
        if "Age_Group" in df.columns:
            age_counts = df['Age_Group'].value_counts()
            st.bar_chart(age_counts)

    st.divider()

    # --- Data Grid ---
    st.subheader("📨 Live Feed & AI Insights")
    
    display_df = df.sort_values("Timestamp", ascending=False).copy()
    
    st.dataframe(
        display_df,
        column_config={
            "Timestamp": st.column_config.TextColumn("Time", width="small"),
            "Name": st.column_config.TextColumn("User Name", width="medium"),
            "Rating": st.column_config.NumberColumn("Stars", format="%d ⭐", width="small"),
            "Review": st.column_config.TextColumn("Customer Review", width="large"),
            "Gender": st.column_config.TextColumn("Gender", width="small"),
            "Age_Group": st.column_config.TextColumn("Age", width="small"),
            "Admin_Summary": st.column_config.TextColumn("AI Summary", width="medium"),
            "Admin_Action": st.column_config.TextColumn("⚡ Recommended Action", width="medium"),
            "User_Response": st.column_config.TextColumn("AI Response Sent", width="large"),
        },
        hide_index=True,
        use_container_width=True
    )
    
    col_btn, _ = st.columns([1, 4])
    if col_btn.button("🔄 Refresh Feed"):
        st.rerun()


# Main Entry Point


def main():
    api_key = FeedbackService.get_api_key()
    if not api_key:
        st.error("🚨 Configuration Error: `GOOGLE_API_KEY` is missing.")
        st.info("Please add it to `.streamlit/secrets.toml`.")
        st.stop()

    with st.sidebar:
        st.header(f"{PAGE_ICON} Navigation")
        selected_view = st.radio(
            "Select Dashboard:", 
            ["User View", "Admin View"],
            captions=["Public Form", "Internal Analytics"]
        )
        st.divider()
        st.caption(f"System Status: ● Online")

    if selected_view == "User View":
        render_user_dashboard(api_key)
    else:
        render_admin_dashboard()

if __name__ == "__main__":
    main()
