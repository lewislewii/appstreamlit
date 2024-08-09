import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import base64

# Load your data
jd_mod = pd.read_csv("sampled_jd_mod.csv")
cv_fin = pd.read_csv("cv_fin.csv")

# Add custom CSS for text shadow, image border, and sidebar background
st.markdown("""
    <style>
    .header-title {
        font-size: 2em;
        # color: #000000; /* Black color */
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.6); /* Shadow effect */
    }
    .header-subtitle {
        font-size: 1.5em;
        # color: #000000; /* Black color */
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.6); /* Shadow effect */
    }
    .logo-image {
        border: 5px solid #000000; /* Black border */
        border-radius: 10px; /* Rounded corners */
        box-shadow: 3px 3px 6px rgba(0, 0, 0, 0.5); /* Shadow effect */
    }
    .sidebar .sidebar-content {
        background-color: #fff; /* Blue background */
        color: #ffffff; /* White text color for contrast */
    }
    .sidebar .sidebar-content a {
        color: #ffffff; /* Ensure link color is white */
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.image("imagesformodel/ima.jpg", width=100)  # Add your logo here

# Apply the custom CSS classes
st.markdown('<p class="header-title">Smart Match</p>', unsafe_allow_html=True)
st.markdown('<p class="header-subtitle">Your Job and Applicant Recommendation System</p>', unsafe_allow_html=True)

# Add background image to the main area (optional)
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
                background-size: cover;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

# Call the function with your image path (optional)
add_bg_from_local("imagesformodel/jobBG.png")

# Streamlit sidebar configuration
st.sidebar.title("Job Recommendation System")
st.sidebar.image("imagesformodel/ima.jpg", width=100)  # Add your logo here

st.sidebar.header("Recommendations")

option = st.sidebar.selectbox("Choose an option", ["Recommend Jobs for Applicant", "Recommend Applicants for Job", "Recommend Jobs for New Applicant"])

if option == "Recommend Jobs for Applicant":
    applicant_option = st.sidebar.selectbox("Select Applicant", [(f"{row['Applicant_ID']} - {row['Current_position']}", row['Applicant_ID']) for _, row in cv_fin.iterrows()])
    applicant_id = applicant_option[1]
    if st.sidebar.button("Recommend Jobs"):
        if applicant_id:
            Job_Applicant(int(applicant_id))
elif option == "Recommend Applicants for Job":
    job_option = st.sidebar.selectbox("Select Job", [(f"{row['Job_ID']} - {row['Job_position']}", row['Job_ID']) for _, row in jd_mod.iterrows()])
    job_id = job_option[1]
    if st.sidebar.button("Recommend Applicants"):
        if job_id:
            Applicant_Job(int(job_id))
elif option == "Recommend Jobs for New Applicant":
    current_jd = st.sidebar.text_area("Enter New Applicant's Job Description:")
    if st.sidebar.button("Recommend Jobs (New Applicant)"):
        if current_jd:
            New_Job_Applicant(current_jd)

# Footer in the main area
st.sidebar.markdown("---")
st.sidebar.info("SmartMatch @2024")
st.sidebar.markdown("[Contact Us](mailto:email@example.com)")
