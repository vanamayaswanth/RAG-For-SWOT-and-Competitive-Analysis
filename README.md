
# Company Analysis Application

## Overview
The Company Analysis Application is a comprehensive tool designed to perform Competitive Intelligence (CI) and SWOT analyses for organizations. It provides actionable insights by processing company-specific data and delivering detailed evaluations. The application ensures high accuracy and efficiency, leveraging state-of-the-art AI and machine learning models.

This project reflects our commitment to utilizing cutting-edge technologies to deliver exceptional value to our clients and stakeholders.

## Key Features
- **Competitive Intelligence Analysis**: Compare two companies to identify trends, competitor strategies, opportunities, and threats.
- **SWOT Analysis**: Evaluate the strengths, weaknesses, opportunities, and threats of a given company.
- **Dynamic Evaluation**: Assess analysis results against specified criteria for correctness, truthfulness, and helpfulness.

## Research and Development
We conducted extensive research to identify industry pain points and ensure the application meets real-world needs. By leveraging insights from data science, AI, and business intelligence domains, we developed a robust solution tailored to our objectives. Our team utilized advanced models, including embeddings and retrieval systems, ensuring the application operates with precision and reliability.

## Technical Details
### Technologies Used:
- **Python 3.12**: Core language for application logic.
- **Streamlit**: Interactive and user-friendly web interface.
- **LangChain**: Chain management and document processing.
- **FAISS**: Vector store for efficient retrieval.
- **Docker**: Containerization for easy deployment and scalability.

### Application Structure:
1. **`app.py`**: Main application file containing the logic for CI and SWOT analyses.
2. **`requirements.txt`**: List of dependencies for the project.
3. **Dockerfile**: Optimized container setup for seamless deployment.

## Installation and Usage
### Prerequisites:
- Docker installed on your system.

### Steps to Run:
1. Clone the repository to your local system:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```
2. Build the Docker image:
   ```bash
   docker build -t company-analysis-app .
   ```
3. Run the Docker container:
   ```bash
   docker run -p 8501:8501 company-analysis-app
   ```
4. Access the application in your browser at `http://localhost:8501`.

## Contribution
This project was developed collaboratively wih AKashda Kadam , emphasizing rigorous testing and iterative improvement. We ensured every module was reviewed and optimized to meet our quality standards.

We welcome suggestions for enhancements to ensure the application continues to align with evolving business requirements. Please reach out through our internal communication channels to share feedback or propose new features.

## Acknowledgments
We extend our gratitude to all team members and stakeholders who contributed to this project. This application is a testament to our collective efforts and our dedication to delivering innovative solutions.
