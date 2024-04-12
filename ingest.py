import os
import transformers
import logging
from torch import cuda, bfloat16
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.retrievers.merger_retriever import MergerRetriever
from langchain.callbacks import StdOutCallbackHandler
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.document_loaders import TextLoader
from Constants import *
import tiktoken
import re

#Global variable
enc = tiktoken.get_encoding("cl100k_base")

def load_single_document(source_location):
  '''Load the single document'''

  print("Function: load_single_document \t Status: Started")
  loader = TextLoader(source_location)
  documents = loader.load()

  print("Function: load_single_document \t Status: Completed")
  return documents


def load_documents(source_location):
  '''Load all documents'''

  print("Function: load_documents \t Status: Started")
  files_list = os.listdir(source_location)
  file_paths = [os.path.join(source_location, single_file) for single_file in files_list]

  number_of_workers =  min(len(file_paths),  os.cpu_count())

  docs = []
  with ProcessPoolExecutor() as executor:
      futures = []
      for file_path in file_paths:
          future = executor.submit(load_single_document, file_path)
          futures.append(future)
      for future in as_completed(futures):
          content = future.result()
          docs.append(content)

  print("Function: load_documents \t Status: Started")
  return docs


def token_length_function(text: str) -> int:
    return len(enc.encode(text))


def split_documents(document, chunk_size):
  '''Split the documents and return the chunks'''

  print("Function: split_documents \t Status: Started")
  # Keeping the chunk_overlap at 25% of the chunk_size
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=(chunk_size/4))
  splits = text_splitter.split_documents(document)

  print("Function: split_documents \t Status: Completed")
  return splits


def embed_documents(device, splits, k):
  '''Generate document embeddings, store in VectoreStore and return retriever'''
  
  print("Function: embed_documents \t Status: Started")
  embed_model_kwargs = {"device":device}
  bge_embed_model_id = "BAAI/bge-base-en"
  bge_embed_encode_kwargs = {'device': device, 'normalize_embeddings': True}
  bge_embed = HuggingFaceEmbeddings(
      model_name=bge_embed_model_id,
      model_kwargs=embed_model_kwargs,
      encode_kwargs=bge_embed_encode_kwargs
  )

  bge_db = Chroma.from_documents(
      documents=splits,
      embedding=bge_embed,
  )

  bge_retriever = bge_db.as_retriever(
    search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.3, "k":k, "include_metadata":True}
  )

  embed_output = {
      "retriever":bge_retriever,
      "bge_db":bge_db,
      "contriever_db":None
  }
  return embed_output  


def get_retriever():
  '''Wrapper function for the retrieval process'''
  
  print("Function: get_retriever \t Status: Started")  

  #Load the documents
  documents = load_documents(source_location)
  doc_splits=[]
  for document in documents:
    #Split the documents into chunks
    chunk_size=800
    doc_splits=split_documents(document, chunk_size)
    
  #Embed documents and create retrievers
  free_gpu_memory()
  embed_output = embed_documents(device, doc_splits, k=3)
  retriever = embed_output['retriever']
  # result_retriever = MergerRetriever(retrievers = [])

  print("Function: get_retriever \t Status: Completed")  
  return retriever