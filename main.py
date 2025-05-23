import os
import re
import json
from typing import Dict, List, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import SearchApiAPIWrapper
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_groq.chat_models import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools.base import BaseTool
from langchain.tools.retriever import create_retriever_tool
from langchain.retrievers.multi_query import MultiQueryRetriever
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check for required environment variables
required_env_vars = ["SEARCHAPI_API_KEY", "GROQ_API_KEY", "DATABASE_URI"]
for var in required_env_vars:
    if not os.getenv(var):
        raise EnvironmentError(f"Missing required environment variable: {var}")

# Set USER_AGENT if not present
USER_AGENT = os.getenv("USER_AGENT", "MultiSourceResearchBot/1.0")

# API keys
SEARCHAPI_API_KEY = os.getenv("SEARCHAPI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize LLM
llm = ChatGroq(model_name="llama3-70b-8192", temperature=0.7)

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Initialize Splitter
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# Initialize SQLDatabase
db = SQLDatabase.from_uri(
    os.getenv("DATABASE_URI", "sqlite:///research_data.db"),
    sample_rows_in_table_info=3
)

# Create SQL agent
sql_toolkit = SQLDatabaseToolkit(db=db, llm=llm)
sql_agent = create_sql_agent(llm=llm, toolkit=sql_toolkit, verbose=False)

# Initialize the search_tool
search_tool = SearchApiAPIWrapper(searchapi_api_key=SEARCHAPI_API_KEY)

# Define files to process (replace with your actual paths)
PDF_FILES = ["concepts.pdf"]  # Ensure these files exist
TEXT_FILES = ["data.txt"]  # Ensure these files exist

# Document processing functions
def process_pdfs(file_paths: List[str]) -> List[Any]:
    """Process PDF files and return chunks."""
    docs = []
    for file_path in file_paths:
        if not os.path.exists(file_path):
            continue
        try:
            loader = PyPDFLoader(file_path)
            loaded_docs = loader.load()
            docs.extend(loaded_docs)
        except Exception:
            continue
    return splitter.split_documents(docs) if docs else []

def process_text_files(file_paths: List[str]) -> List[Any]:
    """Process text files and return chunks."""
    docs = []
    for file_path in file_paths:
        if not os.path.exists(file_path):
            continue
        try:
            loader = TextLoader(file_path, encoding='utf-8')
            try:
                loaded_docs = loader.load()
                docs.extend(loaded_docs)
            except UnicodeDecodeError:
                loader = TextLoader(file_path, encoding='latin-1')
                loaded_docs = loader.load()
                docs.extend(loaded_docs)
        except Exception:
            continue
    try:
        if docs:
            chunks = splitter.split_documents(docs)
            return chunks
        return []
    except Exception:
        return []

def process_web_pages(urls: List[str]) -> List[Any]:
    """Process web pages and return chunks."""
    try:
        loader = WebBaseLoader(urls, requests_per_second=2)
        docs = loader.load()
        return splitter.split_documents(docs)
    except Exception:
        return []

def create_vector_store(documents: List[Any]) -> Optional[FAISS]:
    """Create a FAISS vector store from documents."""
    try:
        if not documents:
            return None
        return FAISS.from_documents(documents, embeddings)
    except Exception:
        return None

def create_faiss_retrieval_tool(vector_store: Optional[FAISS], name: str, description: str) -> Optional[BaseTool]:
    """Create a retrieval tool from a vector store."""
    try:
        if not vector_store:
            return None
        retriever = MultiQueryRetriever.from_llm(
            retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
            llm=llm
        )
        return create_retriever_tool(retriever=retriever, name=name, description=description)
    except Exception:
        return None

class ResearchState(Dict):
    """The state of research process."""
    query: str
    sources: List[str]
    source_results: Dict[str, Any]
    synthesis: Optional[str] = None
    completed: bool = False
    query_analysis: Optional[str] = None

def understand_query(state: ResearchState) -> ResearchState:
    """Understand a query and formulate a research plan."""
    try:
        chain = ChatPromptTemplate.from_messages([
            ("system", "You are a research assistant. Given a query, analyze it and determine what is needed."),
            ("human", "Query: {query}")
        ]) | llm
        response = chain.invoke({"query": state["query"]})
        state["query_analysis"] = response.content or "General research needed"
    except Exception:
        state["query_analysis"] = "General research needed"
    return state

def select_sources(state: ResearchState) -> ResearchState:
    """Select appropriate sources for the query."""
    if "query_analysis" not in state or not state["query_analysis"]:
        state["query_analysis"] = "General research needed"

    try:
        # Modified prompt with clearer JSON instructions
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Based on query analysis, determine which sources would be most relevant. "
                   "Options: web_search, pdf_files, sql_database, text_files. "
                   "Return ONLY a JSON object with no additional text or formatting, like: "
                   "{{'sources': ['web_search', 'pdf_files']}}"),
            ("human", "Query: {query}\nAnalysis: {query_analysis}")
        ])
        
        # First, get the raw response from the LLM
        chain = prompt | llm
        raw_response = chain.invoke({
            "query": state["query"],
            "query_analysis": state["query_analysis"]
        })
        
        # Extract JSON from the response, handling various formats
        response_text = raw_response.content
        
        # Look for JSON patterns in the text
        json_match = re.search(r'\{.*?\}', response_text.replace('\n', ' '), re.DOTALL)
        
        if json_match:
            try:
                json_str = json_match.group(0)
                sources_dict = json.loads(json_str)
                if isinstance(sources_dict, dict) and "sources" in sources_dict:
                    state["sources"] = sources_dict["sources"]
                else:
                    state["sources"] = ["web_search", "pdf_files"]  # Default
            except json.JSONDecodeError:
                state["sources"] = ["web_search", "pdf_files"]  # Default
        else:
            state["sources"] = ["web_search", "pdf_files"]  # Default
            
    except Exception:
        state["sources"] = ["web_search", "pdf_files"]  # Default sources
    
    return state

def gather_information(state: ResearchState) -> ResearchState:
    """Gather information from the selected sources."""
    results = {}
    
    # Initialize source_results if not present
    if "source_results" not in state:
        state["source_results"] = {}
    
    for source in state.get("sources", ["web_search"]):
        try:
            if source == "web_search":
                search_result = search_tool.run(state["query"])
                if isinstance(search_result, str):
                    results["web_search"] = search_result
                elif isinstance(search_result, dict):
                    results["web_search"] = search_result.get("description", "No description available")
                else:
                    results["web_search"] = "Web search returned unexpected format"
            elif source == "pdf_files":
                pdf_docs = process_pdfs(PDF_FILES)
                if pdf_docs:
                    vector_store = create_vector_store(pdf_docs)
                    if vector_store:
                        retriever_tool = create_faiss_retrieval_tool(
                            vector_store,
                            "pdf_retriever",
                            "Retrieve information from PDF files."
                        )
                        if retriever_tool:
                            results["pdf_files"] = retriever_tool.invoke({"input": state["query"]})
                        else:
                            results["pdf_files"] = "Failed to create PDF retriever tool."
                    else:
                        results["pdf_files"] = "Failed to create vector store for PDFs."
                else:
                    results["pdf_files"] = "No content extracted from PDFs."
            elif source == "text_files":
                text_docs = process_text_files(TEXT_FILES)
                if text_docs:
                    vector_store = create_vector_store(text_docs)
                    if vector_store:
                        retriever_tool = create_faiss_retrieval_tool(
                            vector_store,
                            "text_retriever",
                            "Retrieve from text files"
                        )
                        if retriever_tool:
                            results["text_files"] = retriever_tool.invoke({"input": state["query"]})
                        else:
                            results["text_files"] = "Failed to create text retriever tool."
                    else:
                        results["text_files"] = "Failed to create vector store for text files."
                else:
                    results["text_files"] = "No content extracted from text files."
            elif source == "sql_database":
                results["sql_database"] = sql_agent.invoke({
                    "input": f"Based on the query '{state['query']}', extract relevant information from the database."
                })
        except Exception:
            results[source] = "Error processing source"
    
    state["source_results"] = results
    return state

def synthesize_information(state: ResearchState) -> ResearchState:
    """Synthesize the gathered information into a coherent response."""
    if not state.get("source_results"):
        state["synthesis"] = "No information could be gathered from the sources."
        state["completed"] = True
        return state
    
    try:
        chain = ChatPromptTemplate.from_messages([
            ("system", "You are a research assistant. Synthesize the information from various sources into a coherent response."),
            ("human", "Query: {query}\nSource Results: {source_results}")
        ]) | llm
        
        response = chain.invoke({
            "query": state["query"],
            "source_results": state["source_results"]
        })
        
        state["synthesis"] = response.content or "Synthesis failed due to empty response."
        state["completed"] = True
    except Exception:
        state["synthesis"] = "Error synthesizing information"
        state["completed"] = True
    
    return state

def build_graph() -> StateGraph:
    """Build a research workflow graph."""
    workflow = StateGraph(ResearchState)
    
    # Add nodes
    workflow.add_node("understand_query", understand_query)
    workflow.add_node("select_sources", select_sources)
    workflow.add_node("gather_information", gather_information)
    workflow.add_node("synthesize_information", synthesize_information)
    
    # Set entry point
    workflow.set_entry_point("understand_query")
    
    # Add edges
    workflow.add_edge("understand_query", "select_sources")
    workflow.add_edge("select_sources", "gather_information")
    workflow.add_edge("gather_information", "synthesize_information")
    workflow.add_edge("synthesize_information", END)
    
    return workflow.compile()

def research(query: str) -> str:
    """Conduct research on the given query."""
    try:
        # Initialize the graph
        graph = build_graph()
        
        # Create initial state
        initial_state = {
            "query": query,
            "completed": False,
            "sources": [],
            "source_results": {},
            "synthesis": None,
            "query_analysis": None
        }
        
        # Execute the graph
        result = graph.invoke(initial_state)
        
        # Return the synthesized result
        synthesis = result.get("synthesis", "No research results were generated.")
        return synthesis
    except Exception:
        return "Research failed"

# Main execution
if __name__ == "__main__":
    # Check if files exist
    for pdf in PDF_FILES:
        if not os.path.exists(pdf):
            print(f"Warning: PDF file '{pdf}' not found. Update PDF_FILES in the script.")
    
    for txt in TEXT_FILES:
        if not os.path.exists(txt):
            print(f"Warning: Text file '{txt}' not found. Update TEXT_FILES in the script.")
    
    # Main user interface
    print("=== Multi-Source Research Assistant ===")
    print("Enter a research query (or 'exit' to quit):")
    
    while True:
        query = input("> ")
        if query.lower() == "exit":
            break
        
        print("\nResearching... This may take a moment.\n")
        result = research(query)
        print(f"\nResearch Results:\n{result}\n")
