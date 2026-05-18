import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
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

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY is missing")

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=api_key,
)


class FinanceRequest(BaseModel):
    income: str
    expenses: str


def run_finance_agent(income: str, expenses: str):
    expense_analyzer = Agent(
        role="Expense Analyzer",
        goal="Analyze income and expenses clearly",
        backstory="You are a personal finance analyst who studies spending patterns and identifies risky expenses.",
        verbose=True,
        llm=llm,
    )

    budget_planner = Agent(
        role="Budget Planner",
        goal="Create a practical monthly budget",
        backstory="You help users create simple and realistic budgets using needs, wants, savings, and emergency funds.",
        verbose=True,
        llm=llm,
    )

    savings_advisor = Agent(
        role="Savings Advisor",
        goal="Suggest realistic ways to save more money",
        backstory="You give practical money-saving advice that is easy for beginners to follow.",
        verbose=True,
        llm=llm,
    )

    report_writer = Agent(
        role="Finance Report Writer",
        goal="Write a clean beginner-friendly finance report",
        backstory="You convert financial analysis into a simple, structured monthly report.",
        verbose=True,
        llm=llm,
    )

    task1 = Task(
        description=f"""
You are analyzing the user's exact monthly finance data.

STRICT RULES:
- Use ONLY the income provided by the user: ₹{income}
- Use ONLY the expense lines provided by the user.
- Do NOT invent new categories.
- Do NOT change any number.
- Do NOT multiply, round, estimate, or assume missing values.
- If an expense line has no category, call it "Uncategorized".
- Calculate total expenses by adding only the provided expense amounts.
- Calculate remaining money as income minus total expenses.

Income: ₹{income}

Expenses:
{expenses}

Now calculate:
1. Total expenses
2. Remaining money
3. Highest spending categories from the provided list only
4. Risky spending areas from the provided list only
5. Current financial health summary
""",
        expected_output="A clear expense analysis using only the exact user-provided income and expenses.",
        agent=expense_analyzer,
    )

    task2 = Task(
        description=f"""
Create a monthly budget using ONLY this income:

Income: ₹{income}

STRICT RULES:
- Do NOT change the income amount.
- Do NOT invent current expenses.
- Do NOT assume the user earns ₹50,000 or any other amount.
- Use the exact income ₹{income}.
- Base suggestions on the expense analysis from the previous task.
- If using 50/30/20 rule, calculate it from ₹{income} only.

Create a practical budget with:
- Needs
- Wants
- Savings
- Emergency fund
- Education/personal spending if relevant
""",
        expected_output="A practical budget calculated only from the user's exact income.",
        agent=budget_planner,
    )

    task3 = Task(
        description=f"""
Suggest realistic savings ideas based ONLY on the user's provided expenses:

{expenses}

STRICT RULES:
- Do NOT invent categories.
- Do NOT change expense values.
- Do NOT say the user spends more than they entered.
- Give savings suggestions only for categories present in the user's expense list.
- Estimated savings must be realistic and smaller than or equal to the category amount.
""",
        expected_output="Savings suggestions based only on the user's provided expense categories.",
        agent=savings_advisor,
    )

    task4 = Task(
        description=f"""
Write the final monthly personal finance report.

STRICT RULES:
- Monthly income must be exactly: ₹{income}
- Expenses must be exactly from this list:
{expenses}
- Do NOT invent new categories.
- Do NOT change any numbers.
- Do NOT use ₹50,000 unless the user entered ₹50,000.
- Do NOT estimate fake expenses.
- If unsure, say "not provided" instead of inventing data.

Report must include:
1. Monthly income
2. Total expenses
3. Remaining money
4. Current financial health
5. Budget plan
6. Savings suggestions
7. Final action steps
""",
        expected_output="A complete beginner-friendly monthly finance report using only exact user-provided values.",
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