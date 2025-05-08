# Recruitment Agency LangGraph Application

## Overview

This enterprise-grade application leverages Large Language Models (LLMs) with LangGraph to automate and streamline the recruitment process. The system analyzes candidate applications against job requirements, categorizes applicants based on experience level, and generates contextually appropriate response communications.

## Core Features

- **Experience Classification Engine**: Accurately categorizes candidates as Entry, Mid, or Senior level based on comprehensive profile analysis
- **Advanced Skill Matching Algorithm**: Performs detailed evaluation of candidate qualifications against position requirements
- **Intelligent Response System**: Generates personalized, professional communications:
  - HR interview invitations for strong qualification matches
  - Technical assessment invitations for candidates meeting core requirements
  - Professional rejection communications for non-suitable candidates

## Technical Requirements

- Python 3.9+
- OpenAI API key
- Required dependencies (see Installation)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Recruitment-Agency-LangGraph.git
cd Recruitment-Agency-LangGraph
```

2. Set up a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required dependencies:

```bash
pip install -r requirements.txt
```

4. Configure environment variables by creating a `.env` file:

```
OPENAI_API_KEY=your_api_key_here
```

## Usage Guide

Start the application:

```bash
python recruiter.py
```

### Configuration and Customization

To adapt the system for specific recruitment scenarios:

1. Modify the `job_requirements` and `applicant_data` variables in the `main()` function of `recruiter.py`
2. Adjust evaluation parameters in `config.py` as needed
3. Customize response templates for your organization's tone and branding

## System Architecture

- `config.py` - System configuration and state definitions
- `functions.py` - Core processing functions implementing the recruitment workflow logic
- `recruiter.py` - Main application orchestrating the workflow graph

## Implementation Guide

To extend functionality:

1. Implement additional processing functions in `functions.py`
2. Update the workflow graph in the `create_workflow()` method
3. Extend the State TypedDict in `config.py` for any additional data requirements

## Error Management

The application implements robust error handling:
- API authentication validation
- LLM service integration monitoring
- Response format validation
- Comprehensive workflow execution logging

## Performance Considerations

For high-volume recruitment scenarios, consider:
- Implementing batch processing for multiple applications
- Caching common job requirement analyses
- Optimizing LLM prompt engineering for cost efficiency

## License

[MIT License](LICENSE) 
