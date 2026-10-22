import os
import streamlit as st
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from crewai_tools import SerperDevTool
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()

# Streamlit Page Configuration
st.set_page_config(
    page_title="Multi-Agent Support Crew with RAG & Web Search",
    page_icon="🛃",
    layout="centered"
)

st.title("🤖 Multi-Agent RAG & Web Search Support System")
st.markdown("Powered by CrewAI, FAISS Vector DB, Web Search, and Streamlit.")

# Ensure a data directory and sample file exist for the RAG knowledge base
os.makedirs("data", exist_ok=True)
sample_faq_path = "data/faq.txt"
if not os.path.exists(sample_faq_path):
    with open(sample_faq_path, "w", encoding="utf-8") as f:
        f.write(
            "Company FAQ & Policies:\n"
            "- Return Policy: Customers can return any item within 30 days of purchase for a full refund.\n"
            "- Password Reset: To reset your password, click on 'Forgot Password' on the login page and follow the email instructions.\n"
            "- Support Hours: Our customer support team is available 24/7 via live chat and email at support@example.com.\n"
        )

# 1. Build RAG Vector DB (Cached)
@st.cache_resource
def build_vector_db():
    try:
        loader = DirectoryLoader(
            "data",
            glob="**/*.txt",
            loader_cls=TextLoader
        )
        docs = loader.load()
        if not docs:
            return None

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150
        )
        chunks = splitter.split_documents(docs)

        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small"
        )
        vector_db = FAISS.from_documents(chunks, embeddings)
        return vector_db
    except Exception as e:
        st.error(f"Error building Vector DB: {e}")
        return None

vector_db = build_vector_db()

# 2. Create RAG Tool
@tool("Customer Support Knowledge Base")
def customer_support_rag_tool(query: str) -> str:
    """
    Use this tool to search company FAQs, policies, pricing, refund rules,
    product documentation, and customer support knowledge base.
    """
    if vector_db is None:
        return "Knowledge base is currently unavailable."
    results = vector_db.similarity_search(query, k=4)
    context = "\n\n".join([doc.page_content for doc in results])
    return context

# Streamlit UI Input
query = st.text_input("Enter your customer query or task:", placeholder="How do I reset my password?")

if st.button("Run Multi-Agent Crew"):
    if not query.strip():
        st.warning("Please enter a valid query.")
    elif not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY is missing from your environment variables.")
    elif not os.getenv("SERPER_API_KEY"):
        st.error("SERPER_API_KEY is missing from your environment variables.")
    else:
        with st.spinner("Running sequential crew (RAG Assistant -> Web Search -> Entry Agent)..."):
            try:
                llm = LLM(model="gpt-4o-mini")
                search_tool = SerperDevTool()

                # Define the 3 Required Agents
                assistant_agent = Agent(
                    role="Assistant",
                    goal="Answer the user's query directly using the knowledge base RAG tool.",
                    backstory=(
                        "You are a primary assistant equipped with a knowledge base tool. "
                        "You answer customer questions accurately based on company documents."
                    ),
                    tools=[customer_support_rag_tool],
                    llm=llm,
                    verbose=True,
                    allow_delegation=False
                )

                web_search_agent = Agent(
                    role="Web Search Assistant",
                    goal="Search the web for the user's query and generate an answer from the search results.",
                    backstory=(
                        "You are a web search specialist. You look up live, external information "
                        "to provide a thorough perspective on the user's query."
                    ),
                    tools=[search_tool],
                    llm=llm,
                    verbose=True,
                    allow_delegation=False
                )

                entry_agent = Agent(
                    role="Entry Agent",
                    goal="Write the query, Assistant's answer, and Web Search Assistant's answer into a text file named answers.txt, and return both answers.",
                    backstory=(
                        "You are an organized entry agent. You collect the outputs from the previous agents, "
                        "save them cleanly into a local text file called 'answers.txt', and output both answers for the user interface."
                    ),
                    llm=llm,
                    verbose=True,
                    allow_delegation=False
                )

                # Define Sequential Tasks
                task1 = Task(
                    description=f"Search the internal knowledge base using your tool and answer the query: {query}",
                    expected_output="An accurate answer based on the local knowledge base RAG tool.",
                    agent=assistant_agent
                )

                task2 = Task(
                    description=f"Search the web using your web search tool for context on the query: {query}",
                    expected_output="A well-researched answer based on external web search results.",
                    agent=web_search_agent
                )

                task3 = Task(
                    description=(
                        f"Take the original query ('{query}'), the Assistant's answer from Task 1, and the Web Search Assistant's answer from Task 2. "
                        "Write all of them into a local text file named 'answers.txt' with clear formatting. "
                        "Return both answers clearly so they can be presented to the user."
                    ),
                    expected_output="Confirmation that answers.txt was written successfully, along with the text of both answers.",
                    agent=entry_agent
                )

                # Sequential Crew Execution
                crew = Crew(
                    agents=[assistant_agent, web_search_agent, entry_agent],
                    tasks=[task1, task2, task3],
                    process=Process.sequential,
                    verbose=True
                )

                result = crew.kickoff()

                st.success("Crew execution finished successfully!")
                st.markdown("### 📄 Final Results")
                st.write(str(result))

                # Read and show answers.txt contents
                if os.path.exists("answers.txt"):
                    with open("answers.txt", "r", encoding="utf-8") as f:
                        file_content = f.read()
                    with st.expander("View contents of saved `answers.txt`"):
                        st.text(file_content)

            except Exception as e:
                st.error(f"An error occurred: {e}")