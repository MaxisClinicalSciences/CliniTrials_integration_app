import streamlit as st
import requests
import pandas as pd
from urllib.parse import quote_plus  # Import for URL encoding
from langchain import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Set the maximum number of cells allowed for Pandas Styler
pd.set_option("styler.render.max_elements", 800000)  # Adjust this number as necessary

# Function to create a simple login page
def login_page():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type='password')

    if st.button("Login"):
        # Dummy check for authentication
        if username == "admin" and password == "password":  # Replace with actual authentication
            st.session_state["logged_in"] = True
            st.success("Login successful!")
            st.rerun()  # Redirect to the main app
        else:
            st.error("Invalid username or password.")

# Function to check login status
def check_login():
    if "logged_in" not in st.session_state or not st.session_state["logged_in"]:
        login_page()
        st.stop()  # Stop further execution until logged in

# Function to fetch data from ClinicalTrials.gov API
def fetch_data(base_url, conditions):
    combined_data = []  # This will hold the combined data for all selected conditions

    for condition in conditions:
        condition_data = fetch_data_for_condition(base_url, condition)  # Fetch data for each condition
        combined_data.extend(condition_data)  # Append data for this condition to the combined list

    return combined_data

# Function to fetch data for each condition
def fetch_data_for_condition(base_url, condition):
    data_list = []

    params = {
        "pageSize": 100,
        "query.term": quote_plus(condition)  # URL-encode the condition
    }

    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        data = response.json()
        studies = data.get("studies", [])
        for study in studies:
            # Extract relevant information
            nctId = study["protocolSection"]["identificationModule"].get("nctId", "Unknown")
            overallStatus = study["protocolSection"]["statusModule"].get("overallStatus", "Unknown")
            startDate = study["protocolSection"]["statusModule"].get("startDateStruct", {}).get("date", "Unknown Date")
            conditions = ", ".join(study["protocolSection"]["conditionsModule"].get("conditions", ["No conditions listed"]))
            acronym = study["protocolSection"]["identificationModule"].get("acronym", "Unknown")

            interventions_list = study["protocolSection"].get("armsInterventionsModule", {}).get("interventions", [])
            interventions = ", ".join([intervention.get("name", "No intervention name listed") for intervention in interventions_list]) if interventions_list else "No interventions listed"

            locations_list = study["protocolSection"].get("contactsLocationsModule", {}).get("locations", [])
            locations = ", ".join([f"{location.get('city', 'No City')} - {location.get('country', 'No Country')}" for location in locations_list]) if locations_list else "No locations listed"

            primaryCompletionDate = study["protocolSection"]["statusModule"].get("primaryCompletionDateStruct", {}).get("date", "Unknown Date")
            studyFirstPostDate = study["protocolSection"]["statusModule"].get("studyFirstPostDateStruct", {}).get("date", "Unknown Date")
            lastUpdatePostDate = study["protocolSection"]["statusModule"].get("lastUpdatePostDateStruct", {}).get("date", "Unknown Date")
            studyType = study["protocolSection"]["designModule"].get("studyType", "Unknown")
            phases = ", ".join(study["protocolSection"]["designModule"].get("phases", ["Not Available"]))

            # Append the study details to the list
            data_list.append({
                "NCT ID": nctId,
                "Acronym": acronym,
                "Overall Status": overallStatus,
                "Start Date": startDate,
                "Conditions": conditions,
                "Interventions": interventions,
                "Locations": locations,
                "Primary Completion Date": primaryCompletionDate,
                "Study First Post Date": studyFirstPostDate,
                "Last Update Post Date": lastUpdatePostDate,
                "Study Type": studyType,
                "Phases": phases,
            })
    else:
        st.error(f"Failed to fetch data for '{condition}'. Status code: {response.status_code} - {response.text}")

    return data_list

# Function to apply custom styles to the DataFrame
# def style_dataframe(df):
#     numeric_cols = df.select_dtypes(include=['float', 'int']).columns.tolist()

#     styled_df = df.style \
#         .set_table_styles([{
#             'selector': 'thead th',
#             'props': [('background-color', '#007acc'), ('color', 'white')]
#         }]) \
#         .set_properties(**{
#             'border-color': '#007acc',
#             'color': 'black',
#             'background-color': '#f9f9f9'
#         }) \
#         .highlight_null()  # Removed 'null_color' argument for compatibility
#         .format(precision=2)  # This should be properly indented and aligned

#     if numeric_cols:
#         styled_df = styled_df.background_gradient(subset=numeric_cols, cmap='coolwarm')

#     return styled_df

# Check if user is logged in
check_login()

# Function to update Study Type based on Phases and Safety
# Function to update Study Type based on Phases and Safety
def update_study_type(row):
    phases = row.get("Phases", "")
    
    if phases == "NA":
        return "PD"
    
    elif not phases or phases == "Not Available":
        return "PK"
    
    if row["Study Type"] == "INTERVENTIONAL":
        if "PHASE1" in phases or "PHASE 1" in phases:
            return "PK"
        elif any(phase in phases for phase in ["PHASE2", "PHASE 2", "PHASE3", "PHASE 3", "PHASE4", "PHASE 4"]):
            return "PD"
    
    elif row["Study Type"] == "OBSERVATIONAL":
        if "PHASE1" in phases or "PHASE 1" in phases:
            return "PK"
    
    return row["Study Type"]

# Function to update the UI based on selected filter, including Safety
def update_ui_based_on_filter(df):
    study_filter = st.radio("Select Studies", ["All", "Pharmacodynamics (PD)", "Pharmacokinetics (PK)", "Safety"], index=0)

    if study_filter == "Pharmacodynamics (PD)":
        filtered_df = df[df["Study Type"] == "PD"]
    elif study_filter == "Pharmacokinetics (PK)":
        filtered_df = df[df["Study Type"] == "PK"]
    elif study_filter == "Safety":
        filtered_df = df[df["Conditions"].str.contains("safety", case=False, na=False) | 
                         df["Interventions"].str.contains("safety", case=False, na=False)]
    else:
        filtered_df = df

    styled_filtered_df = filtered_df
    st.write(f"### Rows containing '{study_filter}' studies")
    if not filtered_df.empty:
        st.dataframe(styled_filtered_df, use_container_width=True)
    else:
        st.write(f"No '{study_filter}' studies found.")

# Query input and display results based on conditions
def handle_query_input(conditions):
    query = st.text_area(f"Ask a query to the data fetched for {', '.join(conditions)}")

    if query:
        prompt = f"""
        1. Always show answer in tabular format if possible.
        2. Response should not be blank.
        """ + query

        first_input_prompt = PromptTemplate(
            input_variables=[prompt],
            template="Reply this question {first_input_prompt}",
        )

        llm = HuggingFaceHub(
            huggingfacehub_api_token="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",
            repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
            model_kwargs={"temperature": 0.5, "max_new_tokens": 512},
        )

        chain = LLMChain(llm=llm, prompt=first_input_prompt, verbose=False)

        if st.button("Submit Query", type="primary"):
            with st.spinner("Processing your query..."):
                response = chain.run(prompt)

            st.write(response)

# Set page configuration
st.set_page_config(page_title="MCS", layout="wide")

# Set the background and styling
def set_background():
    st.markdown(
        """
        <style>
        .stApp {background-color:;}
        section[data-testid="stSidebar"] {background-color:; border-right: 2px solid #00BFFF;}
        .stButton button {background-color:; color:; border-radius: 10px;}
        .stDataFrame {background-color:; border: 2px solid;}
        </style>
        """, unsafe_allow_html=True
    )

set_background()

# Styling for DataFrame
# def style_dataframe(df):
#     numeric_cols = df.select_dtypes(include=['float', 'int']).columns.tolist()

#     styled_df = df.style \
#         .set_table_styles([{
#             'selector': 'thead th',
#             'props': [('background-color', '#007acc'), ('color', 'white')]
#         }]) \
#         .set_properties(**{'border-color': '#007acc', 'color': 'black', 'background-color': '#f9f9f9'}) \
#         .highlight_null(null_color='#f2f2f2') \
#         .set_precision(2)

#     if numeric_cols:
#         styled_df = styled_df.background_gradient(subset=numeric_cols, cmap='coolwarm')

#     return styled_df

# Application Title
st.title("ClinicalTrials.gov Integration")

# Sidebar for user inputs
st.sidebar.image("MaxisLogo.png")
st.sidebar.title("Fetch Data")

# Multiselect for selecting query titles (conditions) in the sidebar
conditions = st.sidebar.multiselect(
    "Select medical conditions:",
    ["Diabetes", "Asthma", "COVID-19", "Heart Disease", "Asthma in Children", 
     "Breast Cancer", "Alzheimer's in Adults", "Asthma in Adults"],
    default=[]
)

# Button to start fetching data
if st.sidebar.button("Fetch Data"):
    if conditions:
        with st.spinner(f"Fetching data for {', '.join(conditions)}..."):
            base_url = "https://clinicaltrials.gov/api/v2/studies"
            data_list = fetch_data(base_url, conditions)  # Fetch data for all conditions

            # Convert to DataFrame
            if data_list:
                df = pd.DataFrame(data_list)
                df["Study Type"] = df.apply(update_study_type, axis=1)  # Update Study Type based on Phases
                st.session_state.df = df
                st.session_state.data_fetched = True
                st.session_state.conditions = conditions
                st.success("Data fetched successfully.")
                
    else:
        st.error("Please select at least one condition to fetch data.")

# Display fetched data
if "data_fetched" in st.session_state and st.session_state.data_fetched:
    df = st.session_state.df

    st.write("### Fetched Data")
    styled_df = df
    st.dataframe(styled_df, use_container_width=True)
   

    # Optionally, save the DataFrame to a CSV file
    df.to_csv(f"{conditions}_clinical_trials_data.csv", index=False)

    # Count PK and PD studies
    pk_count = df["Study Type"].value_counts().get("PK", 0)
    pd_count = df["Study Type"].value_counts().get("PD", 0)

    # Display the counts
    st.write(f"### Pharmacokinetics Studies Count: {pk_count}")
    st.write(f"### Pharmacodynamics Studies Count: {pd_count}")

    # Filter based on study types or safety
    update_ui_based_on_filter(df)

    # Query input box for further user queries
    handle_query_input(st.session_state.conditions)
