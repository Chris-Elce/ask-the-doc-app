# What is document question-answering?
As the name implies, document question-answering answers questions about a specific
document. This process involves two steps:
Step 1. Ingestion
The document is prepared through a process known as ingestion so that the LLM model can use
it. Ingestion transforms it into an index, the most common being a vector store. The process
involves:
 Loading the document
 Splitting the document
 Creating embeddings
 Storing the embeddings in a database (a vector store)
Step 2. Generation
With the index or vector store in place, you can use the formatted data to generate an answer by
following these steps:
 Accept the user's question
 Identify the most relevant document for the question
 Pass the question and the document as input to the LLM to generate an answer
🦜
Check out the LangChain documentation on question answering over documents.
App overview
At a conceptual level, the app's workflow remains impressively simple:
1. The user uploads a document text file, asks a question, provides an OpenAI API key, and
clicks "Submit."
2. LangChain processes the two input elements. First, it splits the input document into
chunks, creates embedding vectors, and stores them in the embeddings database (i.e., the
vector store). Then it applies the user-provided question to the Question Answering chain
so that the LLM can answer the question:
