import os
import subprocess
import sys

# Function to install a package
def install_package(package):
    try:
        __import__(package)
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install required packages
install_package("langchain")
install_package("openai")
install_package("faiss-cpu")
install_package("pandas")
install_package("plotly")
install_package("numpy")
install_package("scikit-learn")
install_package("matplotlib")
install_package("rich")
install_package("torch")
install_package("transformers")

import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DataFrameLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import pandas as pd

# OpenAI API key (hardcoded)
OPENAI_API_KEY = "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"

# Functions for Titanic data processing and chain building
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

def build_vectorstore(df: pd.DataFrame):
    """Build vectorstore for retrieval."""
    def row_to_text(row):
        survived_text = "survived" if row['Survived'] == 1 else "did not survive"
        return (
            f"Passenger {row['Name']} {survived_text} the Titanic disaster. "
            f"They were a {row['Sex']} passenger aged {row['Age']:.1f} years, "
            f"traveling in {row['Pclass']} class with a fare of ${row['Fare']:.2f}. "
            f"They had {row['SibSp']} siblings/spouses and {row['Parch']} parents/children aboard. "
            f"They embarked from {row['Embarked']} port."
        )

    df["text"] = df.apply(row_to_text, axis=1)
    stats_processor = TitanicDataProcessor(df)
    stats = stats_processor.feature_stats
    global_context = f"""
    The Titanic dataset contains information about {stats['total_passengers']} passengers.
    {stats['total_survivors']} passengers survived the disaster, a survival rate of {stats['survival_rate']:.1f}%.
    """
    
    # Create documents
    loader = DataFrameLoader(df, page_content_column="text")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(documents)
    
    # Create embeddings and vectorstore
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectordb = FAISS.from_documents(split_docs, embedding=embeddings)
    
    return vectordb, stats_processor

def build_qa_chain(vectordb):
    """Build QA chain."""
    template = """
    Use the following context to answer questions. Be concise.
    Context: {context}
    Question: {question}
    Answer:"""
    QA_PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(search_kwargs={"k": 5}),
        chain_type_kwargs={"prompt": QA_PROMPT}
    )
    return qa_chain

# Main Streamlit App
def main():
    st.title("🚢 Titanic Data Analysis Assistant")
    
    try:
        df = pd.read_csv("tested.csv")  # Update the CSV path as needed
        vectordb, data_processor = build_vectorstore(df)
        qa_chain = build_qa_chain(vectordb)
    except Exception as e:
        st.error(f"Error: {e}")
        return
    
    user_input = st.text_input("Ask about the Titanic dataset:")
    if user_input:
        try:
            result = qa_chain({"query": user_input})
            st.write(result["result"])
        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
