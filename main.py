# Load environment variables first
from dotenv import load_dotenv
import os
load_dotenv()


## Initialize a simple Embedding model
from langchain_huggingface import HuggingFaceEmbeddings

embeddings=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
embeddings

#---------------------------------------------------------------------------------------
# Initialize AstraDB vector store with HuggingFace embeddings and connection credentials
#---------------------------------------------------------------------------------------

from langchain_astradb import AstraDBVectorStore

vector_store = AstraDBVectorStore(
    embedding=embeddings,
    api_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT"),
    collection_name="astra_vector_langchain",
    token=os.getenv("ASTRA_DB_TOKEN"),
    namespace=None,
)
vector_store

# vector_store.clear()
# print("✅ All documents deleted successfully from AstraDB collection!")


#---------------------------------------------------------------------------------------
# Define a class for advanced PDF text processing with chunking and cleaning features
#---------------------------------------------------------------------------------------

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List

class SmartPDFProcessor:

    # Initialize the processor with chunk size, overlap, and text splitter configuration
    def __init__(self,chunk_size=1000,chunk_overlap=100):
        self.chunk_size=chunk_size,
        self.chunk_overlap=chunk_overlap,
        self.text_splitter=RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[" "],
        )


    # Process a given PDF file into cleaned and chunked Document objects with metadata
    def process_pdf(self,pdf_path:str)->List[Document]:

        # Load the PDF using LangChain's PyPDFLoader
        loader=PyPDFLoader(pdf_path)
        pages=loader.load()

        # Initialize list to store processed text chunks
        processed_chunks=[]
        for page_num,page in enumerate(pages):                # Iterate through each page for cleaning and chunking
            cleaned_text=self._clean_text(page.page_content)  # Clean extracted text from the page

            if len(cleaned_text.strip()) < 40:                # Skip pages that contain very little text
                continue
            # Split text into chunks and add metadata
            chunks = self.text_splitter.create_documents(
                texts=[cleaned_text],
                metadatas=[{
                    **page.metadata,
                    "page": page_num + 1,
                    "total_pages": len(pages),
                    "chunk_method": "smart_pdf_processor",
                    "char_count": len(cleaned_text)
                }]
            )

            processed_chunks.extend(chunks)                  # Add generated chunks to the final list

        return processed_chunks

    # Clean text by removing extra spaces, fixing ligatures, and unwanted phrases
    def _clean_text(self, text: str) -> str:
        text = " ".join(text.split())                        # Remove excessive whitespace
        text = text.replace("ﬁ", "fi")                       # Fix common PDF extraction issues
        text = text.replace("ﬂ", "fl")

        text = text.replace("Scan to Download", "").strip()  # Remove "Scan to Download" phrase

        return text


preprocessor=SmartPDFProcessor()


try :
    smart_chunks=preprocessor.process_pdf("Never Let Me Go... PDF.pdf")
    print(f"Processed into {len(smart_chunks)} smart chunks")

    # Show enhanced metadata
    if smart_chunks:
        print("\nSample chunk metadata:")
        for key, value in smart_chunks[0].metadata.items():
            print(f"  {key}: {value}")

except Exception as e:
    print(f"Processing error: {e}")


# Insert the cleaned and chunked documents into AstraDB vector store
if smart_chunks:
    inserted_ids = vector_store.add_documents(smart_chunks)
    print(f"\nInserted {len(inserted_ids)} cleaned documents into AstraDB")
else:
    print("⚠️ No chunks to insert into AstraDB.")
    

# Perform the similarity search
result = vector_store.similarity_search("What is Samar studying in college ?", k=5)

for res in result:
    print(f"\n--- Search Result ---")
    print(f"* {res.page_content}\n  [{res.metadata}]")


# Create a retriever from the vector store to fetch top 6 most similar documents during search
retriever = vector_store.as_retriever(
    search_kwargs={"k": 6}
)

# You can now use this retriever object
print(retriever)


import os
from langchain_groq import ChatGroq

# ---------------------------------------
# Initialize LLM (Groq - gemma2-9b-it)
# ---------------------------------------

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

## Import necessary LangChain modules for prompts, output parsing, and runnable operations
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.prompts import ChatPromptTemplate

# ----------------------------------------------------------------------------------------------------
#  Define a custom chat prompt template guiding the model to answer using context or admit ignorance
#-----------------------------------------------------------------------------------------------------

# Create a custom prompt
custom_prompt = ChatPromptTemplate.from_template("""Use the following context to answer the question.
If you don't know the answer based on the context, say you don't know.
Provide specific details from the context to support your answer.

Context:
{context}

Question: {question}

Answer:""")
custom_prompt


# Function to format a list of documents into a single string separated by double newlines
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# ----------------------------------------------------------------------------------------------
# Build a RAG chain using the retriever, custom prompt, and LLM, parsing the output as a string
# ----------------------------------------------------------------------------------------------

rag_chain_lcel=(
    {
        "context":retriever | format_docs,       # Convert retrieved docs to a formatted string
        "question": RunnablePassthrough()        # Pass the question directly
     }
    | custom_prompt           # Apply the prompt template
    | llm                     # Generate answer using the LLM
    | StrOutputParser()       # Parse output as a clean string
)
# Display the constructed RAG chain object
rag_chain_lcel


# ------------------------------------------------------------------------------------------
# Define a function to query the RAG chain with a question and print the result neatly
# ------------------------------------------------------------------------------------------

def query_rag_lcel(question):
    print(f"Question: {question}")
    print("-" * 50)

    answer = rag_chain_lcel.invoke(question)
    print(f"Answer: {answer}")

## Test LCEL chain

print("Testing LCEL Chain:")
query_rag_lcel("What is the name of the girl samar falls in love with ?")

