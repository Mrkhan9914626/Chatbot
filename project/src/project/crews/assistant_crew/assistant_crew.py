from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.memory.storage.ltm_sqlite_storage import LTMSQLiteStorage
from crewai.memory.storage.rag_storage import RAGStorage
from crewai.memory import LongTermMemory, ShortTermMemory, EntityMemory
from crewai_tools import SerperDevTool
from crewai_tools import CodeDocsSearchTool
import os
from dotenv import load_dotenv
from crewai import LLM
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
serper_api_key = os.getenv("SERPER_API_KEY")


llm1 = LLM(
    model=os.getenv("MODEL"),
    api_key=api_key,
)


search_tools_1 = SerperDevTool()
search_tools_2 = CodeDocsSearchTool( config=dict(
        llm=dict(
            provider="google", # or google, openai, anthropic, llama2, ...
            config=dict(
                model="gemini/gemini-1.5-flash",
                api_key=api_key,
                # temperature=0.5,
                # top_p=1,
                # stream=true,
            ),
        ),
        embedder=dict(
            provider="google", # or openai, ollama, ...
            config=dict(
                model="models/text-embedding-004",
                task_type="retrieval_document",
                # title="Embeddings",
            ),
        ),
    )
)

@CrewBase
class AssistantCrew:
    """Assistant Crew"""


    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"


    @agent
    def personal_assistant(self) -> Agent:
        return Agent(
            config=self.agents_config["personal_assistant"],
            llm=llm1,
            tools=[search_tools_1, search_tools_2],
            # verbose=True  
        )


    @task
    def task_handler(self) -> Task:
        return Task(
            config=self.tasks_config["task_handler"],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Assistant Crew"""


        return Crew(
            agents=self.agents, 
            tasks=self.tasks,  
            process=Process.sequential,
            # verbose=True,
            memory=True,
            long_term_memory=LongTermMemory(
                storage=LTMSQLiteStorage(
                    db_path=".content/long_term_memory.db",
                )
            ),
            short_term_memory=ShortTermMemory(
                storage=RAGStorage(
                    embedder_config={
                        "provider": "google",
                        "config":{
                        "model": "models/text-embedding-004",
                        "api_key": api_key,}
                    },
                    type="short_term",
                    path=".content/short_term_memory.db",
                )
            ),
            entity_memory=EntityMemory(
                storage=RAGStorage(
                    embedder_config={
                        "provider": "google",
                        "config":{
                        "model": "models/text-embedding-004",
                        "api_key": api_key,}
                    },
                    type="entity",
                    path=".content/entity_memory.db",
                )
            ),
        )
    