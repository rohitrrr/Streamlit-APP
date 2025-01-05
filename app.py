import os
import streamlit as st
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DataFrameLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re

class TitanicDataProcessor:
    def __init__(self, df: pd.DataFrame):
        self.raw_df = df.copy()
        self.processed_df = None
        self.feature_stats = {}
        self.process_data()
        
    def process_data(self):
        df = self.raw_df.copy()
        
        # Handle missing values
        df['Age'] = df['Age'].fillna(df['Age'].median())
        df['Fare'] = df['Fare'].fillna(df['Fare'].median())
        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
        
        # Calculate and store basic statistics
        self.feature_stats = {
            'total_passengers': len(df),
            'total_survivors': df['Survived'].sum(),
            'male_survivors': df[df['Sex'] == 'male']['Survived'].sum(),
            'female_survivors': df[df['Sex'] == 'female']['Survived'].sum(),
            'survival_rate': df['Survived'].mean() * 100,
            'male_survival_rate': (df[df['Sex'] == 'male']['Survived'].mean() * 100),
            'female_survival_rate': (df[df['Sex'] == 'female']['Survived'].mean() * 100),
            'avg_age': df['Age'].mean(),
            'avg_fare': df['Fare'].mean()
        }
        
        self.processed_df = df

def build_vectorstore(df: pd.DataFrame, openai_api_key: str):
    """Enhanced vectorstore building with better context."""
    
    def row_to_text(row):
        survived_text = "survived" if row['Survived'] == 1 else "did not survive"
        return (
            f"Passenger {row['Name']} {survived_text} the Titanic disaster. "
            f"They were a {row['Sex']} passenger aged {row['Age']:.1f} years, "
            f"traveling in {row['Pclass']} class with a fare of ${row['Fare']:.2f}. "
            f"They had {row['SibSp']} siblings/spouses and {row['Parch']} parents/children aboard. "
            f"They embarked from {row['Embarked']} port."
        )

    # Create text representation
    df["text"] = df.apply(row_to_text, axis=1)
    
    # Add global statistics
    stats_processor = TitanicDataProcessor(df)
    stats = stats_processor.feature_stats
    
    global_context = f"""
    The Titanic dataset contains information about {stats['total_passengers']} passengers.
    {stats['total_survivors']} passengers survived the disaster, a survival rate of {stats['survival_rate']:.1f}%.
    {stats['male_survivors']} males and {stats['female_survivors']} females survived.
    The male survival rate was {stats['male_survival_rate']:.1f}% while the female survival rate was {stats['female_survival_rate']:.1f}%.
    The average passenger age was {stats['avg_age']:.1f} years and the average fare was ${stats['avg_fare']:.2f}.
    """
    
    # Add global context as a document
    docs = [{"text": global_context}]
    df_docs = df.to_dict('records')
    docs.extend(df_docs)
    
    # Create documents
    loader = DataFrameLoader(pd.DataFrame(docs), page_content_column="text")
    documents = loader.load()
    
    # Split documents
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(documents)
    
    # Create embeddings and vectorstore
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectordb = FAISS.from_documents(split_docs, embedding=embeddings)
    
    return vectordb, stats_processor

def build_qa_chain(vectordb, openai_api_key):
    """Enhanced QA chain with better prompting."""
    
    template = """
    You are an expert data analyst focusing on the Titanic dataset. Use the following pieces of context to answer the question. If you don't know the answer, say that you don't know, don't try to make up an answer.

    Context: {context}

    Question: {question}

    When answering:
    1. If the question asks for numerical statistics, provide exact numbers from the data
    2. If the question asks about survival rates, provide percentages
    3. If relevant, mention both absolute numbers and percentages
    4. Be concise but complete in your answer

    Answer:"""

    QA_PROMPT = PromptTemplate(
        template=template, input_variables=["context", "question"]
    )

    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        model_name="gpt-3.5-turbo",
        temperature=0,
        max_tokens=512
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(search_kwargs={"k": 5}),
        chain_type_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True
    )
    
    return qa_chain

def main():
    st.title("🚢 Titanic Data Analysis Assistant")
    
    # Load environment variables
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not openai_api_key:
        st.error("OpenAI API key not found. Please set OPENAI_API_KEY in your .env file.")
        st.stop()
    
    # Load data
    try:
        df = pd.read_csv(r"tested.csv")  # Update with your path
        vectordb, data_processor = build_vectorstore(df, openai_api_key)
        qa_chain = build_qa_chain(vectordb, openai_api_key)
    except Exception as e:
        st.error(f"Error initializing the system: {e}")
        return
    
    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if isinstance(message["content"], (go.Figure, px.Figure)):
                st.plotly_chart(message["content"])
            else:
                st.markdown(message["content"])
    
    user_input = st.chat_input("Ask about the Titanic dataset...")
    if user_input:
        st.chat_message("user").markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        try:
            with st.spinner("Analyzing..."):
                result = qa_chain({"query": user_input})
                answer = result["result"]
                
                # Check if visualization is needed
                if any(keyword in user_input.lower() for keyword in ['distribution', 'plot', 'chart', 'graph', 'show']):
                    fig = None
                    if 'age' in user_input.lower() and 'distribution' in user_input.lower():
                        fig = px.histogram(data_processor.processed_df, x='Age', 
                                         color='Survived', 
                                         title='Age Distribution by Survival Status')
                    elif 'fare' in user_input.lower() and 'distribution' in user_input.lower():
                        fig = px.box(data_processor.processed_df, x='Survived', 
                                   y='Fare', 
                                   title='Fare Distribution by Survival Status')
                    
                    if fig:
                        st.session_state.messages.append({"role": "assistant", "content": fig})
                        with st.chat_message("assistant"):
                            st.plotly_chart(fig)
                            st.markdown(answer)
                    else:
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        with st.chat_message("assistant"):
                            st.markdown(answer)
                else:
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    with st.chat_message("assistant"):
                        st.markdown(answer)
                
        except Exception as e:
            error_msg = f"Error processing your question: {str(e)}"
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            st.chat_message("assistant").markdown(error_msg)

if __name__ == "__main__":
    main()
