from dotenv import load_dotenv
from langchain_groq.chat_models import ChatGroq
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
# from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.globals import set_debug, set_verbose

set_debug(True)        # Most detailed logs
# or
# set_verbose(True)    # Slightly less noisy

load_dotenv()

# instantialing llm
llm = ChatGroq(model="qwen/qwen3-32b", reasoning_effort="none")

# creating a supervisor node
def supervisor_node(input):
    '''This is a supervisor node that reviews the input and provides guidance'''
    # defining prompt
    supervisor_prompt = ChatPromptTemplate.from_messages(
        [
            ("system","You are a supervisor node in a langgraph based system. Please review the input by user and provide guidance."),
            ("human", "{input}")
        ]
    )

    # invoking llm with the prompt 
    supervisor_chain = supervisor_prompt | llm
    supervisor_response = supervisor_chain.invoke({"input": input})
    return supervisor_response



# intialisng embdding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embedding_model)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

def rag_node(query):
    '''This is a RAG node that retrieves relevant information from the indexed chunks based on the query and provides an answer'''
    relevant_chunks = retriever.invoke(query)
    # defining prompt
    rag_prompt = ChatPromptTemplate.from_messages(
        [
            ("system","Answer the user query only from the context provided in the relevant chunks. If the relevant chunks do not contain the answer, say 'I don't know.'"),
            ("human", "Query: {query}\nRelevant Chunks: {relevant_chunks}")
        ]
    )

    # invoking llm with the prompt
    rag_chain = rag_prompt | llm
    rag_response = rag_chain.invoke({"query": query, "relevant_chunks": relevant_chunks })
    return rag_response
    


if __name__ == "__main__":
    # result = rag_node("What is BERT model in deep learning?")
    result = rag_node("What is BERT stands for?")
    print(result.content)