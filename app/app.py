import datetime
import json
import logging
import os
import pickle

import joblib
import pandas as pd
from flask import Flask, request
from flask_cors import CORS
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.llm import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import PromptTemplate
from langchain.output_parsers import OutputFixingParser

# convert to dict
def el_convert_to_dict(explain_local, prediction, simple=False):
    # A local explanation shows the breakdown of how much each term contributed to 
    # the prediction for a single sample. The intercept reflects the average case. 
    # In regression, the intercept is the average y-value of the train set 
    # (e.g., $5.51 if predicting cost). In classification, the intercept is the 
    # log of the base rate (e.g., -2.3 if the base rate is 10%). The 15 most 
    # important terms are shown.
  
    # simple only keep without &
  
    data = explain_local.data(0)
    explanation = {}

    # intercept = data['extra']['scores'][0]

    for i in range(len(data['names'])):
        if simple:
            if '&' in data['names'][i]:
                continue
        explanation[data['names'][i]] = data['scores'][i]

    return {
        # 'call_for_interview': data['perf']['predicted'],
        'call_for_interview': prediction[0],
        'explanation': explanation
    }

def predict(df):
    filename = os.path.join(os.path.dirname(__file__), "ebm_model.pkl")

    # load the model from disk
    ebm = pickle.load(open(filename, 'rb'))

    # predict for single data point
    new_data = df

    prediction = ebm.predict(new_data)

    # explainable local, new_data is the same as above
    explain_local = ebm.explain_local(new_data)

    # display the explanation, print(explain_local.data) is also possible
    data = el_convert_to_dict(explain_local, prediction, simple=True)

    return data

# from secure import require_apikey

logger = logging.getLogger()

app = Flask(__name__)

CORS(app)

app.config.from_object(__name__)
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))


@app.route('/health')
def health():
    return 'It is alive!\n'

resume_analysis_template = """<|im_start|>system
You are an assessor tasked with evaluating a candidate's resume for a specific job role. Provide a detailed analysis on the following aspects:

- Years of experience: Calculate the number of years the candidate has spent working in the relevant field, based on the information in their resume, and compare it to the job requirements.
- Functional competency: Assess the extent to which the candidate's skills, knowledge, and abilities in their resume align with the functional aspects of the job role.
- Top functional skill: Identify the most important functional skill required for the job and evaluate the candidate's proficiency in that skill.
- Second most important functional skill: Determine the second most critical functional skill needed for the job and evaluate the candidate's proficiency in that skill.
- Third most important functional skill: Ascertain the third most essential functional skill necessary for the role and assess the candidate's proficiency in that skill.
- Behavioral competency: Evaluate the candidate's behavioral skills or soft skills as mentioned in their resume, and examine how well they align with the job role requirements.
- Top behavioral skill: Identify the most important behavioral skill necessary for the job and gauge the candidate's proficiency in that area.
- Second most important behavioral skill: Determine the second most crucial behavioral skill needed for the job and evaluate the candidate's proficiency in that skill.
- Third most important behavioral skill: Ascertain the third most essential behavioral skill required for the role and assess the candidate's proficiency in that skill.

Current date: {current_date}

<|im_end|>
<|im_start|>user
■ Resume ■
{resume}

■ Job Requirements ■
{job_description}

<|im_end|>
<|im_start|>assistant
"""

RESUME_ANALYSIS_PROMPT = PromptTemplate(
    template=resume_analysis_template, 
    input_variables=["resume", "job_description", "current_date"],
)

resume_score_template = """<|im_start|>system
Assistant is given with detailed assessment of candidate's resume. Give scores, you always have to output scores. Current date: {current_date}

{format_instructions}

<|im_end|>
<|im_start|>user
■ Detailed assessment ■
{assessment}

<|im_end|>
<|im_start|>assistant
"""

response_schemas = [
    ResponseSchema(name="years_of_experience", description="The total number of years the candidate.", type="integer"),
    ResponseSchema(name="years_of_experience_score", description="A score (1-10) representing the candidate's years of experience.", type="integer"),
    ResponseSchema(name="functional_competency_score", description="A composite score (1-10) evaluating the candidate's proficiency in all required functional skills.", type="integer"),
    ResponseSchema(name="top1_skills_score", description="A score (1-10) assessing the candidate's aptitude in the most critical functional skill for the job, determined.", type="integer"),
    ResponseSchema(name="top2_skills_score", description="A score (1-10) evaluating the candidate's competence in the second most important functional skill for the job.", type="integer"),
    ResponseSchema(name="top3_skills_score", description="A score (1-10) gauging the candidate's proficiency in the third most significant functional skill for the job.", type="integer"),
    ResponseSchema(name="behavior_competency_score", description="An aggregated score (1-10) reflecting the candidate's overall performance in required behavioral competencies.", type="integer"),
    ResponseSchema(name="top1_behavior_skill_score", description="A score (1-10) rating the candidate's mastery of the foremost required behavioral skill.", type="integer"),
    ResponseSchema(name="top2_behavior_skill_score", description="A score (1-10) appraising the candidate's aptitude in the second most vital behavioral skill.", type="integer"),
    ResponseSchema(name="top3_behavior_skill_score", description="A score (1-10) measuring the candidate's ability in the third most essential behavioral skill.", type="integer"),
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

RESUME_SCORE_PROMPT = PromptTemplate(
    template=resume_score_template, 
    input_variables=["assessment", "current_date"],
    partial_variables={"format_instructions": format_instructions}
)

write_email_template = """<|im_start|>system
You are an HR assistant responsible for drafting emails to candidates about their interview status. You have access to a detailed analysis of their resume, the final decision on their application, and an explanation of the various factors that influenced the decision. Your email should:

Politely greet the candidate.
Express gratitude for their application.
Succinctly provide information on the final decision regarding their interview invitation.
If they are not selected, kindly use the explanation provided to outline the factors that contributed to the decision in a clear and constructive manner.
Additionally, provide personalized feedback in an ordered bullet list (e.g., 1., 2., 3.) with suggestions on how they can improve their chances for future opportunities based on their resume analysis and the provided explanation.
If they are selected, provide necessary details, such as the date and time of the interview, along with any additional instructions.
Thank the candidate again and close the email professionally.

Please maintain a respectful tone and provide helpful feedback that is tailored to the candidate's application, considering the explanation and resume analysis provided. Present this guidance in an organized, easy-to-read manner with ordered bullet points, enabling them to understand their areas of improvement and increase their chances of success in future applications.

Current date: {current_date}

<|im_end|>
<|im_start|>user
■ Resume analysis ■
{resume_analysis}

■ Final Decision ■
Call for interview = {call_for_interview} 

■ Explanations ■
```json
{explanations}
```

■ Candidate Name ■
{candidate_name}

■ Company Name ■
{company_name}

<|im_end|>
<|im_start|>assistant
"""

WRITE_EMAIL_PROMPT = PromptTemplate(
    template=write_email_template, 
    input_variables=["resume_analysis", "candidate_name", "company_name", "call_for_interview", "explanations", "current_date"],
)

# @require_apikey

@app.route('/explain', methods=['POST'])
def explain():
    body = request.get_json(force=True)
    # messages_array = body.get('messages') or []

    resume = body.get('resume')
    job_description = body.get('position')
    candidate_name = body.get('candidate_name') or 'Moiz Farooq'
    company_name = body.get('company_name') or 'App4HR'

    current_date = datetime.datetime.now().strftime("%Y-%m-%d")

    llm = ChatOpenAI(temperature=0)
    llm_chain = LLMChain(
        llm=llm,
        prompt=RESUME_ANALYSIS_PROMPT,
        verbose=True,
    )

    assessment = llm_chain.run(resume=resume, job_description=job_description, current_date=current_date)

    llm = ChatOpenAI(temperature=0)
    llm_chain = LLMChain(
        llm=llm,
        prompt=RESUME_SCORE_PROMPT,
        verbose=True,
    )

    result = llm_chain.run(assessment=assessment, current_date=current_date)
    
    final_result = None

    try:
        final_result = output_parser.parse(result)

    except:
        new_parser = OutputFixingParser.from_llm(parser=output_parser, llm=ChatOpenAI())
        final_result = new_parser.parse(result)
    
    for key in final_result:
        if final_result[key] is None:
            final_result[key] = 0

    years_of_experience = None
    if 'years_of_experience_score' in final_result:
        years_of_experience = final_result['years_of_experience']
        final_result['years_of_experience'] = final_result['years_of_experience_score']
        del final_result['years_of_experience_score']

    df = pd.json_normalize(final_result)

    # "2scaler.joblib" is in same directory as this file
    scaler_filename = os.path.join(os.path.dirname(__file__), "2scaler.joblib")
    scaler = joblib.load(scaler_filename)

    normalized_new_data = scaler.transform(df)

    columns = df.columns
    index = df.index
    df_normalized = pd.DataFrame(data=normalized_new_data, columns=columns, index=index)

    result = predict(df_normalized)

    call_for_interview = "YES" if result['call_for_interview'] == 1 else "NO"
    explanations = json.dumps(result['explanation'])

    llm = ChatOpenAI(temperature=0)
    llm_chain = LLMChain(
        llm=llm,
        prompt=WRITE_EMAIL_PROMPT,
        verbose=True,
    )

    email = llm_chain.run(resume_analysis=assessment, call_for_interview=call_for_interview, explanations=explanations, company_name=company_name, candidate_name=candidate_name, current_date=current_date)

    return {
        'call_for_interview': 1 if call_for_interview == "YES" else 0,
        'explanations': result['explanation'],
        'email': email
    }


@app.route('/dummy')
def dummy():
    return {
        'text': 'Hello World!',
    }, 200
