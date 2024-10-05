import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_openai import OpenAI

# Load environment variables
load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

# # Load and process the text
# file_path = r"Text.txt"
# loader = TextLoader(file_path)
# documents = loader.load()

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=80, chunk_overlap=20)
# doc_splitted = text_splitter.split_documents(documents)

# Create embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever()

# Define system message and create chat prompt
system_message = """You are a helpful and professional chatbot for Hygienewunder, 
a German-based company specializing in hygiene and cleaning products. 
Your primary role is to assist customers with their inquiries and facilitate 
the sales process by:

Product Knowledge: Providing accurate and detailed information about Hygienewunder's products. 
Always extract relevant information from the official product details available at 
[https://www.hygienewunder.at/info/]. Ensure that you have comprehensive knowledge of 
each product's features, ingredients, and uses.

Product Recommendations: Based on user input (such as pet ownership, allergies, or specific 
cleaning needs), recommend one of the four available product packages. Engage users by asking 
clarifying questions to tailor your recommendations effectively.

For example, if a user mentions having pets, suggest the appropriate product package designed 
for pet owners.

Automated Sales Process: After making a recommendation, guide users through the purchasing 
process by providing a direct payment link (such as Stripe) to complete their order. Ensure 
that users understand the purchasing steps clearly.

No Discounts: Under no circumstances should you offer or mention any discounts, promotions, 
or special offers. The company does not authorize any discounts.

When communicating with customers:
- Be clear, polite, and concise in your responses.
- Use formal language in German (Sie) when responding to inquiries in German.
- Tailor your responses based on customer input, always ensuring that the information you 
  provide is helpful and user-friendly.
- Avoid making assumptions or providing information that deviates from the official product details."""

system_template = system_message + "\n\nContext: {context}"
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

human_template = "User: {question}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([
    system_message_prompt,
    human_message_prompt
])

# Create LLM and conversation chain
llm = OpenAI(temperature=0.4)
conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    combine_docs_chain_kwargs={"prompt": chat_prompt}
)

# Streamlit app
st.title("Hygienewunder Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is your question?"):
    st.chat_message("user").markdown(prompt)  # Display user message in chat message container
    st.session_state.messages.append({"role": "user", "content": prompt})  # Add user message to chat history
    response = conversation_chain({"question": prompt, "chat_history": []})
    with st.chat_message("assistant"):  # Display assistant response in chat message container
        st.markdown(response['answer'])
    st.session_state.messages.append({"role": "assistant", "content": response['answer']})  # Add assistant response to chat history