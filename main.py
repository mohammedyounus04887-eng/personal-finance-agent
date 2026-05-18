import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from crewai import Agent, Task, Crew, Process, LLM
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="AI Personal Finance Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

llm = LLM(
    model="openrouter/openai/gpt-4o-mini",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENAI_API_KEY"),
)

class FinanceRequest(BaseModel):
    income: str
    expenses: str

def run_finance_agent(income: str, expenses: str):
    expense_analyzer = Agent(
        role="Expense Analyzer",
        goal="Analyze income and expenses clearly",
        backstory="You are a personal finance analyst.",
        verbose=True,
        llm=llm,
    )

    budget_planner = Agent(
        role="Budget Planner",
        goal="Create a practical monthly budget",
        backstory="You help users create simple and realistic budgets.",
        verbose=True,
        llm=llm,
    )

    savings_advisor = Agent(
        role="Savings Advisor",
        goal="Suggest realistic ways to save more money",
        backstory="You give practical money-saving advice.",
        verbose=True,
        llm=llm,
    )

    report_writer = Agent(
        role="Finance Report Writer",
        goal="Write a clean beginner-friendly finance report",
        backstory="You convert financial analysis into a simple report.",
        verbose=True,
        llm=llm,
    )

    task1 = Task(
        description=f"""
Analyze this monthly finance data.

Income: ₹{income}

Expenses:
{expenses}

Calculate:
1. Total expenses
2. Remaining money
3. Highest spending categories
4. Risky spending areas
5. Financial health summary
""",
        expected_output="A clear expense analysis with totals and comments.",
        agent=expense_analyzer,
    )

    task2 = Task(
        description=f"""
Using the income ₹{income}, create a better monthly budget.

Use simple categories:
- Needs
- Wants
- Savings
- Emergency fund
- Education
- Personal spending

Also use the 50/30/20 rule where suitable.
""",
        expected_output="A practical monthly budget with suggested amounts.",
        agent=budget_planner,
    )

    task3 = Task(
        description="""
Suggest realistic ways to save money.

Focus on:
- Food spending
- Shopping
- Subscriptions
- Travel
- Entertainment

Give estimated savings where possible.
""",
        expected_output="A list of savings suggestions with estimated savings.",
        agent=savings_advisor,
    )

    task4 = Task(
        description="""
Create the final personal finance report.

Include:
- Monthly income
- Total expenses
- Current financial health
- Budget plan
- Savings suggestions
- Final action steps
""",
        expected_output="A complete beginner-friendly monthly finance report.",
        agent=report_writer,
    )

    crew = Crew(
        agents=[
            expense_analyzer,
            budget_planner,
            savings_advisor,
            report_writer,
        ],
        tasks=[
            task1,
            task2,
            task3,
            task4,
        ],
        process=Process.sequential,
        verbose=True,
    )

    return str(crew.kickoff())

@app.get("/")
def home():
    return {
        "message": "AI Personal Finance Agent is running",
        "docs": "Go to /docs to test the API",
    }

@app.post("/analyze")
def analyze(data: FinanceRequest):
    try:
        report = run_finance_agent(data.income, data.expenses)
        return {"report": report}
    except Exception as e:
        return {"error": str(e)}