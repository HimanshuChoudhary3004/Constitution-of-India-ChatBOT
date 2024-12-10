import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain


# upload pdf file 
st.header('My chatbot')

with st.sidebar:
    st.title('Your documents')
    file = st.file_uploader('upload a pdf file and start asking questions',type='pdf')
    
# Extract the text
if file is not None:
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
        # st.write(text)
        
# Break it into chunks
    text_splitter = RecursiveCharacterTextSplitter(
                                            separators='\n',
                                            chunk_size = 1000,
                                            chunk_overlap = 150,
                                            length_function = len,
                                            )

    chunks = text_splitter.split_text(text)
    #st.write(chunks)


    # Generating embeddings
    embeddings = OpenAIEmbeddings(openai_api_key = XXXXXXXXXXX)
    # creating vector store
    vector_store = FAISS.from_texts(chunks,embeddings)


    # Get user question
    user_question = st.text_input("Type your question here ...")
    
    # Do similarity check
    if user_question:
        match = vector_store.similarity_search(user_question)
        #st.write(match)


    # Generate question answer chain
    llm = ChatOpenAI(
                    openai_api_key = OPENAI_API_KEY,
                    temperature=0.5,
                    max_tokens=3000,
                    model="gpt-3.5-turbo")
    chain = load_qa_chain(llm=llm,chain_type="stuff")
    response = chain.run(input_documents = match, question = user_question)
    st.write(response)
