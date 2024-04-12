import os
import gc
import json
import transformers
import ingest
from Constants import *
from torch import cuda, bfloat16
from prompts import *
from langchain import PromptTemplate
from langchain.chains import RetrievalQA, ConversationalRetrievalChain, LLMChain
from langchain.document_transformers import (
    EmbeddingsRedundantFilter,
    EmbeddingsClusteringFilter,
)
from langchain.llms import HuggingFacePipeline
from GPUtil import showUtilization as gpu_usage
from langchain.vectorstores import Chroma, FAISS
from langchain.document_loaders import TextLoader
from langchain.callbacks import StdOutCallbackHandler
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain.document_transformers import LongContextReorder
from langchain.retrievers import ContextualCompressionRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

class Query_Rewriter():
    def __init__(self, llm_pipeline, memory, query_rewrite_chain):
        self.llm_pipeline = llm_pipeline
        self.memory = memory
        self.query_rewrite_chain = query_rewrite_chain
    
    def rewrite_query(self, query):
        free_gpu_memory()
        response = self.query_rewrite_chain(query)['text']
        return response



    
