from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

def main():
    print("Hello from langchain-course!")

    information = """
    Alexander Chuka Iwobi MON (/ɪˈwoʊbi/ ih-WOH-bee;[3] born 3 May 1996) is a Nigerian professional footballer who plays as a midfielder for Premier League club Fulham and the Nigeria national team.

Iwobi began his career at Arsenal, with whom he won the FA Cup in 2017 and finished as runner-up for the EFL Cup in 2018 and UEFA Europa League in 2019. In 2019, Iwobi transferred to Everton, and in 2023 he joined Fulham.

Iwobi represented England up to under-18 level. He made his senior international debut for Nigeria in October 2015, and was part of their squads at the FIFA World Cup in 2018 and the Africa Cup of Nations in 2019 (finishing third), 2021, 2023 (being runner-up) and 2025 (finishing third).
    """
    
    summary_template = """
    given the information {information} about the person, i want you to create:
    1. a short summary
    2. two interesting facts about the person
    """
    summary_prompt_template = PromptTemplate(
        template=summary_template,
        input_variables=["information"]
    )
    summary_chain = summary_prompt_template | ChatGroq(model="openai/gpt-oss-120b", temperature=0)
    summary = summary_chain.invoke({"information": information})

    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
    chain = summary_prompt_template | llm

    response = chain.invoke({"information": information})
    print(response.content)

if __name__ == "__main__":
    main()
