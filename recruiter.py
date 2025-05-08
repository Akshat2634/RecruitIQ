import logging
import asyncio
from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from config import State, ConfigError
from functions import (
    categorize_experience,
    assess_skillset,
    schedule_hr_interview,
    schedule_technical_interview,
    reject_applicant,
    route_func,
)

# Configure logging
logger = logging.getLogger(__name__)


class RecruitmentWorkflow:
    """
    Recruitment workflow manager that handles the application assessment process.
    """

    def __init__(self):
        """Initialize the recruitment workflow."""
        self.workflow = None
        self.compiled_app = None

    async def create_workflow(self) -> StateGraph:
        """
        Create the workflow graph with nodes and edges.

        Returns:
            StateGraph: The configured workflow graph
        """
        logger.info("Creating recruitment workflow")

        # Create the workflow
        workflow = StateGraph(State)

        # Add nodes
        workflow.add_node("categorize_experience", categorize_experience)
        workflow.add_node("assess_skillset", assess_skillset)
        workflow.add_node("schedule_hr_interview", schedule_hr_interview)
        workflow.add_node("schedule_technical_interview", schedule_technical_interview)
        workflow.add_node("reject_applicant", reject_applicant)

        # Add edges between nodes
        workflow.add_edge(START, "categorize_experience")
        workflow.add_edge("categorize_experience", "assess_skillset")
        workflow.add_conditional_edges("assess_skillset", route_func)
        workflow.add_edge("schedule_hr_interview", END)
        workflow.add_edge("schedule_technical_interview", END)
        workflow.add_edge("reject_applicant", END)

        self.workflow = workflow
        return workflow

    def compile_workflow(self) -> None:
        """Compile the workflow for execution."""
        if not self.workflow:
            raise ValueError("Workflow not created. Call create_workflow first.")

        logger.info("Compiling recruitment workflow")
        self.compiled_app = self.workflow.compile()

    async def process_application(
        self, applicant_data: str, job_requirements: str
    ) -> Dict[str, Any]:
        """
        Process a job application through the workflow.

        Args:
            applicant_data: The applicant's resume/application data
            job_requirements: The job requirements

        Returns:
            Dict[str, Any]: The final state with the response

        Raises:
            ValueError: If the workflow is not compiled
        """
        if not self.compiled_app:
            raise ValueError("Workflow not compiled. Call compile first.")

        logger.info("Processing new job application")

        # Initialize the state
        initial_state = {
            "applicant": applicant_data,
            "job_requirements": job_requirements,
            "experience_level": "",
            "skill_match": "",
            "response": "",
            "error": None,
        }

        try:
            # Run the workflow
            result = await self.compiled_app.ainvoke(initial_state)

            if result.get("error"):
                logger.error(f"Workflow completed with error: {result['error']}")
            else:
                logger.info("Workflow completed successfully")

            return result
        except Exception as e:
            logger.error(f"Error processing application: {str(e)}")
            return {
                "applicant": applicant_data,
                "job_requirements": job_requirements,
                "experience_level": "",
                "skill_match": "",
                "response": "",
                "error": f"Failed to process application: {str(e)}",
            }


async def main() -> None:
    """Main entry point for the recruitment workflow."""
    try:
        # Configure logging format
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Sample job requirements and applicant data
        job_requirements = """
        Required Skills:
        - Python programming (3+ years)
        - Experience with AI/ML frameworks
        - Strong problem-solving abilities
        - Team collaboration experience
        """

        applicant_data = """
        Name: John Doe
        Experience: 4 years as Python Developer
        Skills: Python, TensorFlow, PyTorch, SQL
        Projects: Developed ML models for customer segmentation
        """

        # Create and initialize the workflow
        recruitment_workflow = RecruitmentWorkflow()
        await recruitment_workflow.create_workflow()
        recruitment_workflow.compile_workflow()

        # Process the application
        result = await recruitment_workflow.process_application(
            applicant_data=applicant_data, job_requirements=job_requirements
        )

        # Check for errors
        if result.get("error"):
            logger.error(f"Application processing failed: {result['error']}")
            print(f"Error: {result['error']}")
        else:
            # Print the result
            print("\n=== Application Processing Result ===")
            print(f"Experience Level: {result['experience_level']}")
            print(f"Skill Match: {result['skill_match']}")
            print("\n=== Response Email ===\n")
            print(result["response"])

    except ConfigError as e:
        logger.critical(f"Configuration error: {e}")
        print(f"Configuration error: {e}")
    except Exception as e:
        logger.critical(f"Unexpected error: {e}")
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    asyncio.run(main())
