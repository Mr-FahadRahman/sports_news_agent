import time
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv

load_dotenv()

def safe_run(agent, prompt, retries=3, wait_time=5):
    for _ in range(retries):
        try:
            return agent.run(prompt)
        except Exception as e:
            if "rate_limit_exceeded" in str(e):
                print(f"Rate limit reached. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise  # Rethrow the error if it's not a rate limit issue
    print("Failed after multiple retries.")
    return None


# Web Agent: Fetches real-time news using DuckDuckGo
web_agent = Agent(
    name="Web News Agent",
    role="Fetch real-time news",
    model=Groq(id="deepseek-r1-distill-llama-70b"),
    tools=[DuckDuckGo()],  # DuckDuckGo tool for live search
    instructions=[
        "Use DuckDuckGo to find the latest ICC Champions Trophy 2025 news.",
        "Return only the top 5 most relevant updates with source links."
    ],
    show_tool_calls=True,
    markdown=True
)

# Sports Agent: Summarizes and analyzes sports news
sports_agent = Agent(
    name="Sports Analyst",
    role="Summarize and analyze ICC Champions Trophy 2025 news",
    model=Groq(id="deepseek-r1-distill-llama-70b"),
    instructions=[
        "Summarize the latest ICC Champions Trophy 2025 news provided by the Web News Agent.",
        "Highlight key takeaways, team updates, and expert insights."
    ],
    show_tool_calls=True,
    markdown=True
)

# **Agent Team: Works together**
agent_team = Agent(
    name="ICC News Team",
    model=Groq(id="deepseek-r1-distill-llama-70b"),
    team=[web_agent, sports_agent],  # Both agents working as a team
    instructions=[
        "The Web News Agent must fetch the latest ICC Champions Trophy 2025 news.",
        "The Sports Analyst must summarize and analyze the news from the Web News Agent.",
        "Ensure the information is accurate, relevant, and well-structured."
    ],
    show_tool_calls=True,
    markdown=True
)

# **Fetch news and analyze**
# news_response = agent_team.run("Get the latest top 5 news updates on ICC Champions Trophy 2025.")
# news_response = safe_run(agent_team, "Get the latest top 5 news updates on ICC Champions Trophy 2025.")
news_response = safe_run(agent_team, "What is the score of today's India vs Newzealand match?")



# **Print final output**
print("\n📰 Latest ICC Champions Trophy 2025 News:\n")
print(news_response.content)


