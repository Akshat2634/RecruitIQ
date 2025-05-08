import streamlit as st
import asyncio
import logging
from recruiter import RecruitmentWorkflow
from config import ConfigError, extract_text_from_pdf

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="AI Recruitment Assistant",
    page_icon="👔",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Create a header
st.title("AI Recruitment Assistant")
st.markdown("### Automated Job Application Processing System")

# Create sidebar with information
with st.sidebar:
    st.header("About")
    st.info(
        """
        This application uses AI to process job applications and determine:
        
        1. Experience Level (Entry, Mid, Senior)
        2. Skill Match (Match, Partial-Match, No-Match)
        3. Appropriate response (Interview invitation or rejection)
        
        The system uses LangChain and LLM to analyze applications.
        """
    )

    st.header("How to Use")
    st.markdown(
        """
        1. Enter the job requirements
        2. Enter the applicant's resume/CV or upload a PDF
        3. Click "Process Application"
        4. View the results and generated response
        """
    )

# Create two columns for input
col1, col2 = st.columns(2)

with col1:
    st.subheader("Job Requirements")
    job_requirements = st.text_area(
        "Enter the job requirements:",
        height=300,
        placeholder="""
        Required Skills:
        - Python programming (3+ years)
        - Experience with AI/ML frameworks
        - Strong problem-solving abilities
        - Team collaboration experience
        """,
    )

with col2:
    st.subheader("Applicant Information")
    
    # Create tabs for text input and PDF upload
    input_tab1, input_tab2 = st.tabs(["Text Input", "PDF Upload"])
    
    with input_tab1:
        applicant_data = st.text_area(
            "Enter the applicant's resume/CV:",
            height=300,
            placeholder="""
            Name: John Doe
            Experience: 4 years as Python Developer
            Skills: Python, TensorFlow, PyTorch, SQL
            Projects: Developed ML models for customer segmentation
            """,
        )
    
    with input_tab2:
        uploaded_file = st.file_uploader("Upload applicant's resume (PDF)", type="pdf")
        if uploaded_file is not None:
            try:
                with st.spinner("Extracting text from PDF..."):
                    applicant_data = extract_text_from_pdf(uploaded_file)
                st.success("PDF successfully processed!")
                st.expander("Preview extracted text").write(applicant_data[:500] + "..." if len(applicant_data) > 500 else applicant_data)
            except ConfigError as e:
                st.error(f"Error processing PDF: {e}")
                applicant_data = ""
            except Exception as e:
                st.error(f"Unexpected error processing PDF: {e}")
                logger.error(f"PDF processing error: {str(e)}")
                applicant_data = ""

# Process button
if st.button("Process Application", type="primary", use_container_width=True):
    if not job_requirements or not applicant_data:
        st.error("Please provide both job requirements and applicant information.")
    else:
        try:
            with st.spinner("Processing application... This may take a moment."):
                # Create progress bar
                progress_bar = st.progress(0)

                # Create and initialize the workflow
                recruitment_workflow = RecruitmentWorkflow()

                # Use asyncio to run the async functions
                async def process():
                    progress_bar.progress(25)
                    await recruitment_workflow.create_workflow()
                    progress_bar.progress(50)
                    recruitment_workflow.compile_workflow()
                    progress_bar.progress(75)
                    result = await recruitment_workflow.process_application(
                        applicant_data=applicant_data, job_requirements=job_requirements
                    )
                    progress_bar.progress(100)
                    return result

                # Run the async function
                result = asyncio.run(process())

                # Display results
                if result.get("error"):
                    st.error(f"Application processing failed: {result['error']}")
                else:
                    # Create tabs for different views of the results
                    tab1, tab2 = st.tabs(["Summary", "Response Email"])

                    with tab1:
                        st.success("Application processed successfully!")

                        # Create metrics
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Experience Level", result["experience_level"])
                        with col2:
                            st.metric("Skill Match", result["skill_match"])

                        # Determine outcome based on skill match
                        if result["skill_match"] == "Match":
                            st.info("Outcome: Invited for HR Interview")
                        elif result["skill_match"] == "Partial-Match":
                            st.info("Outcome: Invited for Technical Interview")
                        else:
                            st.info("Outcome: Application Rejected")

                    with tab2:
                        st.subheader("Generated Response Email")
                        st.text_area(
                            "Email Content",
                            value=result["response"],
                            height=400,
                            disabled=True,
                        )

                        # Add a download button for the email
                        st.download_button(
                            label="Download Email",
                            data=result["response"],
                            file_name="applicant_response.txt",
                            mime="text/plain",
                        )

        except ConfigError as e:
            st.error(f"Configuration error: {e}")
            st.info("Make sure you have set up your OpenAI API key in the .env file.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            logger.error(f"UI error: {str(e)}")

# Add footer
st.markdown("---")
st.markdown("AI Recruitment Assistant | Powered by LangChain and OpenAI")
