import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load your data
jd_mod = pd.read_csv(r"sampled_jd_mod.csv")
cv_fin = pd.read_csv(r"data\cv_fin.csv")

# Check if the necessary columns exist in the dataframes
if 'jd_lem' not in jd_mod.columns or 'Current_jd' not in cv_fin.columns:
    st.error("Necessary columns are missing in the dataframes.")
else:
    # Load your model if it's saved externally, otherwise define your vectorizer
    tfidf_vect = TfidfVectorizer(min_df=5, max_df=0.95)
    tfidf_vect.fit(jd_mod["jd_lem"])

    # Vectorize applicants' job experience
    cv_tfidf = tfidf_vect.transform(cv_fin["Current_jd"])

    def Applicant_Job(job_id):
        if job_id in jd_mod["Job_ID"].tolist():
            index = np.where(jd_mod["Job_ID"] == job_id)[0]
            jd_q = jd_mod.iloc[index[0]:(index[-1] + 1)]

            st.write(f"Information about Vacancy: {job_id}")
            st.dataframe(jd_q)

            jd_tfidf = tfidf_vect.transform(jd_q["Job_description"])
            similarity_score = [linear_kernel(jd_tfidf, cv_tfidf[i])[0][0] for i in range(cv_tfidf.shape[0])]
            top_indices = sorted(range(len(similarity_score)), key=lambda i: similarity_score[i], reverse=True)[:10]
            
            recommendation = pd.DataFrame({
                "Job_ID": [job_id] * 10,
                "Recommended_Applicant_ID": [cv_fin["Applicant_ID"].iloc[i] for i in top_indices]
            })

            nearest_candidates = recommendation["Recommended_Applicant_ID"]
            applicant_recommended = pd.DataFrame(columns=["Job_ID", "Job_position", "Recommended_Applicant_ID", "Work_experience", "Previous_job"])

            for count, applicant_id in enumerate(nearest_candidates):
                index_resume = cv_fin.index[cv_fin["Applicant_ID"] == applicant_id][0]
                job_position = jd_mod[jd_mod["Job_ID"] == job_id]["Job_position"].iloc[0]
                work_experience = cv_fin["Current_jd"].iloc[index_resume]
                previous_job = cv_fin["Current_position"].iloc[index_resume]
                
                applicant_recommended.loc[count] = [job_id, job_position, applicant_id, work_experience, previous_job]
            
            st.write(f"\nRecommended Applicant IDs for Vacancy {job_id}\n")
            st.dataframe(applicant_recommended)
            
            return applicant_recommended
        else:
            st.write("This Job_ID is not in the Jobs' list")
            return None

    def Job_Applicant(applicant_id):
        if applicant_id not in cv_fin["Applicant_ID"].tolist():
            st.write("This Applicant_ID is not in Applicants' list")
            return None
        
        index = np.where(cv_fin["Applicant_ID"] == applicant_id)[0]
        cv_q = cv_fin.iloc[index[0]:(index[-1] + 1)]

        st.write(f"Information about Applicant: {applicant_id}")
        st.dataframe(cv_q)

        cv_tfidf = tfidf_vect.transform(cv_q["Current_jd"])
        jd_tfidf = tfidf_vect.transform(jd_mod["Job_description"])
        similarity_scores = [linear_kernel(cv_tfidf, jd_tfidf[i])[0][0] for i in range(jd_tfidf.shape[0])]

        if len(cv_q) > 1:
            st.write("\nThis Applicant has more than 1 resume (different job description)\n")
        else:
            st.write("\nThis Applicant has only 1 resume\n")
        
        top_indices = sorted(range(len(similarity_scores)), key=lambda i: similarity_scores[i], reverse=True)[:10]
        recommendation = pd.DataFrame({
            "Applicant_ID": [applicant_id] * 10,
            "Recommended_Job_ID": [jd_mod["Job_ID"].iloc[i] for i in top_indices]
        })

        job_recommended = pd.DataFrame(columns=["Applicant_ID", "Applicant_job_title", "Recommended_Job_ID", "Job_description", "Job_title"])

        for count, job_id in enumerate(recommendation["Recommended_Job_ID"]):
            index_vacancy = jd_mod.index[jd_mod["Job_ID"] == job_id][0]
            applicant_job_title = cv_fin["Current_position"].iloc[index[0]:(index[-1] + 1)].tolist()
            job_description = jd_mod["Job_description"].iloc[index_vacancy]
            job_title = jd_mod["Job_position"].iloc[index_vacancy]
            
            job_recommended.loc[count] = [applicant_id, applicant_job_title, job_id, job_description, job_title]
        
        st.dataframe(job_recommended.head())

        return job_recommended



    # Streamlit UI
    st.title("Job Recommendation System")

    # Select recommendation type
    option = st.selectbox("Choose an option", ["Recommend Jobs for Applicant", "Recommend Applicants for Job"])

    if option == "Recommend Jobs for Applicant":
        applicant_option = st.selectbox("Select Applicant", [(f"{row['Applicant_ID']} - {row['Current_position']}", row['Applicant_ID']) for _, row in cv_fin.iterrows()])
        applicant_id = applicant_option[1]
        if st.button("Recommend Jobs"):
            if applicant_id:
                Job_Applicant(int(applicant_id))
    elif option == "Recommend Applicants for Job":
        job_option = st.selectbox("Select Job", [(f"{row['Job_ID']} - {row['Job_position']}", row['Job_ID']) for _, row in jd_mod.iterrows()])
        job_id = job_option[1]
        if st.button("Recommend Applicants"):
            if job_id:
                Applicant_Job(int(job_id))
