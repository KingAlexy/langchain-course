from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch



llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
tools = [TavilySearch()]
agent = create_agent(model=llm,tools=tools)

def main():
    print("hello from alex langchain course by 3.39am after afiama's birthday")
    result = agent.invoke({"messages":HumanMessage(content="What is the weather in Uyo today?")})
    print(result)

if __name__ == "__main__":
    main()