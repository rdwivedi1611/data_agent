import os
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool, initialize_agent
from langchain.agents import AgentType
from cleaner import DataCleaner
from reporter import Reporter

# Make sure your OpenAI API key is set
# export OPENAI_API_KEY="your_key"

class LLMDataAgent:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.cleaned_df = None
        self.report = None
        self.llm = ChatOpenAI(temperature=0)

        # Define tools for the agent
        self.tools = [
            Tool(
                name="Data Cleaner",
                func=self.clean_data_tool,
                description="Use this tool to clean missing values, encode categories, and scale numeric data."
            ),
            Tool(
                name="Report Generator",
                func=self.generate_report_tool,
                description="Generates a CSV report of cleaning steps."
            ),
        ]

        # Initialize LangChain agent
        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors="Check"
        )

    def clean_data_tool(self, action: str) -> str:
        """
        Tool function for LangChain. Accepts a string action (like 'fill missing Age with mean')
        """
        cleaner = DataCleaner(self.df)
        self.cleaned_df, self.report = cleaner.run_all()
        return "Data cleaned successfully."

    def generate_report_tool(self, action: str) -> str:
        reporter = Reporter(self.cleaned_df, self.report)
        reporter.generate_report()
        self.cleaned_df.to_csv("cleaned_dataset.csv", index=False)
        return "Cleaning report and cleaned dataset saved."

    def chat(self, user_input: str) -> str:
        """
        Pass user input to the LangChain agent and get response
        """
        response = self.agent.run(user_input)
        return response
