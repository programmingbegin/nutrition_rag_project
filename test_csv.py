import pandas as pd
from pathlib import Path
from langchain_core.documents import Document
import os
from dotenv import load_dotenv

# folder = Path(r"C:\Users\dprad\Documents\Nutrition RAG Project\RAG_Data")
# def load_excel_documents():
#     "Load every excel file"
#     docs=[]

#     for file in folder.glob("*.csv"):
#         print(file.stem)
#         df = pd.read_csv(file)
#         for idx,row in df.iterrows():
#             text = ", ".join(f"{col}: {row[col]}" for col in df.columns)
#             print(text)
#             docs.append(Document(page_content=text,
#             metadata={"source": file.stem, "row": idx, "type": "csv"}))      
#     return docs  


load_dotenv()
print(os.getenv("OPENAI_API_KEY"))