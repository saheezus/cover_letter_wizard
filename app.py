from flask import Flask, render_template, request
from os import environ
from dotenv import load_dotenv

from llama_index.core import (
    Document,
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)

from llama_index.core.schema import TextNode

from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI

import phoenix as px

from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from phoenix.otel import register

app = Flask(__name__)

# Route for the form page
@app.route('/')
def form():
    return render_template('form.html')

# Route to handle form submission
@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        resume_input = request.form['text_input']
        description_input = request.form['text_input_2']

        answer = submit_prompt(resume_input, description_input)
        return answer
    
def submit_prompt(resume, desc):
    llm = OpenAI(model="gpt-4")
    session = px.launch_app()
    tracer_provider = register()
    LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)

    resume_doc = Document(text=resume)
    desc_doc = Document(text=desc)

    # build index
    resume_index = VectorStoreIndex.from_documents([resume_doc], show_progress=True)
    desc_index = VectorStoreIndex.from_documents([desc_doc], show_progress=True)

    resume_index.storage_context.persist(persist_dir="./storage/resume")
    desc_index.storage_context.persist(persist_dir="./storage/job_description")

    # persist index
    
    resume_engine = resume_index.as_query_engine(similarity_top_k=3, llm=llm)
    desc_engine = desc_index.as_query_engine(similarity_top_k=3, llm=llm)

    query_engine_tools = [
        QueryEngineTool(
            query_engine=resume_engine,
            metadata=ToolMetadata(
                name="resume_engine",
                description=(
                    "Summarize experiences and technical skills from this resume"
                    "Use a detailed plain text question as input to the tool."
                ),
            ),
        ),
        QueryEngineTool(
            query_engine=desc_engine,
            metadata=ToolMetadata(
                name="desc_engine",
                description=(
                    "Provide specific requirements and preferred qualifications for this job posting"
                    "Use a detailed plain text question as input to the tool."
                ),
            ),
        ),
    ]

    cover_letter_template = """
    Could you please write a personalized cover letter based on the resume and job description?
    Cover Letter Template Instructions:

    Greeting: Start with a formal greeting. If a hiring manager's name is known, use it; otherwise, "Dear Hiring Manager" is appropriate.

    Introduction:

    Introduce yourself briefly.
    Mention the position you are applying for and the company name.
    Explain why you are excited about the role and how it aligns with your career goals or aspirations.
    First Paragraph (Experience and Skills):

    Highlight key relevant experiences and skills that match the job description.
    Emphasize technical and non-technical skills (e.g., problem-solving, project management).
    Mention specific projects or achievements that demonstrate your abilities.
    Second Paragraph (Relevance to Company):

    Show enthusiasm for the company's mission or industry, even if it's new to you.
    Describe how your background can be valuable to the company.
    Mention your willingness to learn quickly and adapt to new environments if applicable.
    Conclusion:

    Reiterate your interest in the position.
    Express your eagerness to contribute to the company's goals and mission.
    Mention your availability for further discussions or interviews.
    Closing: Use a polite closing statement such as "Thank you for considering my application."

    Sign off with "Sincerely" or "Best regards."
    Include your name and any relevant contact information (e.g., LinkedIn profile or personal website).

    """
    agent = ReActAgent.from_tools(
        query_engine_tools,
        llm=llm,
        verbose=True,
        max_turns=10,
    )

    response = agent.chat(cover_letter_template)
    return str(response)


if __name__ == '__main__':
    app.run(debug=True)