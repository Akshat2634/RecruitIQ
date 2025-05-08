from config import State, create_llm, ConfigError, logger
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Create LLM instance at module level for reuse
try:
    llm = create_llm()
except ConfigError as e:
    logger.critical(f"Failed to initialize LLM: {e}")
    raise

async def categorize_experience(state: State) -> State:
    """
    Categorize the applicant's experience level based on their application.

    Args:
        state: Current workflow state

    Returns:
        State: Updated workflow state with experience_level

    Raises:
        Exception: If LLM invocation fails
    """
    logger.info("Categorizing experience level of applicant")
    try:
        prompt = ChatPromptTemplate.from_template(
            """
            Based on the following job application, categorize the applicant's experience level into one of the following categories:
            - Entry Level: 0-2 years of relevant experience, limited professional projects, recent graduate
            - Mid Level: 3-5 years of experience, demonstrated expertise in specific areas, led small projects
            - Senior Level: 6+ years of experience, extensive project leadership, deep technical expertise, mentorship experience
            
            Response Format: [Entry-Level, Mid-Level, Senior-Level]

            Application:
            {application}
            """
        )

        chain = prompt | llm | StrOutputParser()

        experience_level = await chain.ainvoke({"application": state["applicant"]})
        logger.info(f"Experience level determined: {experience_level}")

        state["experience_level"] = experience_level
        return state
    except Exception as e:
        logger.error(f"Error in categorize_experience: {str(e)}")
        state["error"] = f"Failed to categorize experience: {str(e)}"
        return state


async def assess_skillset(state: State) -> State:
    """
    Assess the applicant's skillset against job requirements.

    Args:
        state: Current workflow state

    Returns:
        State: Updated workflow state with skill_match

    Raises:
        Exception: If LLM invocation fails
    """
    logger.info("Assessing skillset of applicant")
    try:
        prompt = ChatPromptTemplate.from_template(
            """
            Based on the following job application, experience level and job requirements, assess the applicant's skillset and match it against the job requirements:
        
            - Match: The applicant's skills match the job requirements
            - Partial Match: The applicant's skills partially match the job requirements
            - No Match: The applicant's skills do not match the job requirements 
            
            Response Format: [Match, Partial-Match, No-Match]

            Experience Level: {experience_level}

            Job Requirements:
            {job_requirements}

            Application:
            {application}
            """
        )

        chain = prompt | llm | StrOutputParser()

        skill_match = await chain.ainvoke(
            {
                "job_requirements": state["job_requirements"],
                "application": state["applicant"],
                "experience_level": state["experience_level"],
            }
        )
        logger.info(f"Skill match determined: {skill_match}")

        state["skill_match"] = skill_match
        return state
    except Exception as e:
        logger.error(f"Error in assess_skillset: {str(e)}")
        state["error"] = f"Failed to assess skillset: {str(e)}"
        return state


async def schedule_hr_interview(state: State) -> State:
    """
    Generate an HR interview invitation email.

    Args:
        state: Current workflow state

    Returns:
        State: Updated workflow state with response email

    Raises:
        Exception: If LLM invocation fails
    """
    logger.info("Generating HR interview invitation")
    try:
        prompt = ChatPromptTemplate.from_template(
            """
            Based on the following job application, write a congratulatory email to the applicant:
            
            Write a warm and enthusiastic email congratulating the applicant on passing the initial screening.
            Mention that their qualifications have impressed the hiring team and they are being invited for an HR interview.
            
            Include details about their experience level and how their skills align with the job requirements.
            
            Response Format:
            - A professional congratulatory email with HR interview invitation
            
            Experience Level: {experience_level}
            Skill Match: {skill_match}
            Job Requirements: {job_requirements}
            Application: {application}
            """
        )

        chain = prompt | llm | StrOutputParser()

        response = await chain.ainvoke(
            {
                "experience_level": state["experience_level"],
                "skill_match": state["skill_match"],
                "job_requirements": state["job_requirements"],
                "application": state["applicant"],
            }
        )

        logger.info("HR interview invitation email generated successfully")
        state["response"] = response
        return state
    except Exception as e:
        logger.error(f"Error in schedule_hr_interview: {str(e)}")
        state["error"] = f"Failed to generate HR interview invitation: {str(e)}"
        return state


async def schedule_technical_interview(state: State) -> State:
    """
    Generate a technical interview invitation email.

    Args:
        state: Current workflow state

    Returns:
        State: Updated workflow state with response email

    Raises:
        Exception: If LLM invocation fails
    """
    logger.info("Generating technical interview invitation")
    try:
        prompt = ChatPromptTemplate.from_template(
            """
            Based on the following job application, write a professional email to the applicant inviting them for a technical interview:
            
            Write a clear and professional email inviting the applicant for a technical interview.
            Mention that their skills show potential alignment with the position requirements and the technical team would like to further assess their capabilities.
            
            Include details about their experience level and specific technical skills that caught your attention.
            
            Response Format:
            - A professional email with technical interview invitation
            
            Experience Level: {experience_level}
            Skill Match: {skill_match}
            Job Requirements: {job_requirements}
            Application: {applicant}
            """
        )

        chain = prompt | llm | StrOutputParser()

        response = await chain.ainvoke(
            {
                "experience_level": state["experience_level"],
                "skill_match": state["skill_match"],
                "job_requirements": state["job_requirements"],
                "applicant": state["applicant"],
            }
        )

        logger.info("Technical interview invitation email generated successfully")
        state["response"] = response
        return state
    except Exception as e:
        logger.error(f"Error in schedule_technical_interview: {str(e)}")
        state["error"] = f"Failed to generate technical interview invitation: {str(e)}"
        return state


async def reject_applicant(state: State) -> State:
    """
    Generate a rejection email for the applicant.

    Args:
        state: Current workflow state

    Returns:
        State: Updated workflow state with rejection email

    Raises:
        Exception: If LLM invocation fails
    """
    logger.info("Generating applicant rejection email")
    try:
        prompt = ChatPromptTemplate.from_template(
            """
            Based on the following job application, write a polite rejection letter:

            Experience Level: {experience_level}
            Skill Match: {skill_match}
            Job Requirements: {job_requirements}
            Application: {application}

            Your rejection should be professional, respectful, and provide a brief explanation
            based on the mismatch between the applicant's qualifications and the job requirements.
            """
        )

        chain = prompt | llm | StrOutputParser()

        response = await chain.ainvoke(
            {
                "experience_level": state["experience_level"],
                "skill_match": state["skill_match"],
                "job_requirements": state["job_requirements"],
                "application": state["applicant"],
            }
        )

        logger.info("Rejection email generated successfully")
        state["response"] = response
        return state
    except Exception as e:
        logger.error(f"Error in reject_applicant: {str(e)}")
        state["error"] = f"Failed to generate rejection email: {str(e)}"
        return state


async def route_func(state: State) -> str:
    """
    Determine the next node in the workflow based on skill match.

    Args:
        state: Current workflow state

    Returns:
        str: Name of the next node in the workflow
    """
    logger.info(f"Routing based on skill match: {state['skill_match']}")

    if "error" in state and state["error"]:
        logger.error(f"Workflow encountered an error: {state['error']}")
        return "reject_applicant"

    if state["skill_match"] == "Match":
        return "schedule_hr_interview"
    elif state["skill_match"] == "Partial-Match":
        return "schedule_technical_interview"
    else:
        return "reject_applicant"
