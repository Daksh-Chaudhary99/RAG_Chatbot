import os
import gc
import json
import transformers
import ingest
from prompts import *
from Constants import *
from query_rewriter import Query_Rewriter
from torch import cuda, bfloat16
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


def get_model():
  '''Load the LLM'''

  print("Function: get_model \t Status: Started")
  model = transformers.AutoModelForCausalLM.from_pretrained(
      model_path,
      quantization_config=bnb_config,
      device_map='auto',
      use_auth_token=hf_auth
  )
  model.eval()
  print(f"Model loaded on {device}")

  print("Function: get_model \t Status: Completed")
  return model


def get_llm_pipeline(model):
  '''Create the transformer LLM pipeline with the loaded model'''

  print("Function: get_llm_pipeline \t Status: Started")

  # Loading the Llama2 tokenizer
  tokenizer = transformers.AutoTokenizer.from_pretrained(
      model_path,
      use_auth_token=hf_auth
  )

  # Transformer pipeline parameters
  transformer_pipeline = transformers.pipeline(
      model=model,
      tokenizer=tokenizer,
      return_full_text=True,
      task='text-generation',
      temperature=0.0,
      max_length=4096,
      repetition_penalty=1.15 
  )

  # Creating the LLM pipeline
  llm = HuggingFacePipeline(pipeline= transformer_pipeline)

  print("Function: get_llm_pipeline \t Status: Completed")
  return llm


def get_prompt():
  '''Return the prompt template'''

  print("Function: get_prompt \t Status: Started")

  # Prompt template that follows Llama2 model's training procedure (https://huggingface.co/blog/llama2#how-to-prompt-llama-2)
  template="""<s>[INST] <<SYS>>
    You are a chatbot. Your role is to assist users with their queries based on the given context. Do not rely on external knowledge sources. Ensure that your answer is complete, accurate, and precise as per the given context documents.
    If a user's question is unclear or factually incoherent, provide a clear explanation instead of attempting to answer incorrectly.
    <</SYS>>
    **Context:**
    {context}

    **User Question:**
    {question}

    **Answer:[/INST]**"""

  print("Function: get_prompt \t Status: Completed")
  return template


def get_memory():
  '''Initialize the conversation memory'''

  print("Function: get_memory \t Status: Started")
  memory = ConversationBufferWindowMemory(k=1, input_key="question", memory_key="chat_history")

  print("Function: get_memory \t Status: Completed")
  return memory


def get_query_rewrite_chain(llm, memory):
    '''Get the Query rewriting chain with the loaded LLM pipeline'''

    print("Function: get_query_rewrite_chain \t Status: Started")
    query_rewrite_prompt = PromptTemplate(template=query_rewrite_template, input_variables=['chat_history', 'question'])
    query_rewrite_chain = LLMChain(llm=llm, prompt=query_rewrite_prompt, memory=memory, verbose=True)

    print("Function: get_query_rewrite_chain \t Status: Completed")
    return query_rewrite_chain


def get_RAG_pipeline(llm, retriever, prompt_template, memory):
  '''Initialize the Retrieval Augmented Generation pipeline'''

  print("Function: get_RAG_pipeline \t Status: Started")
  reordering = LongContextReorder()
  pipeline = DocumentCompressorPipeline(transformers=[reordering])

  reordered_retriever = ContextualCompressionRetriever(
      base_compressor=pipeline, base_retriever=retriever
  )

  rag_pipeline = RetrievalQA.from_chain_type(
      llm=llm,
      chain_type='stuff',
      retriever=reordered_retriever,
      return_source_documents=True,
      chain_type_kwargs={
          "prompt": PromptTemplate(
              template = prompt_template,
              input_variables = ["context", "question"]
          ),
          "memory": memory
          },
      verbose=True,
  )

  print("Function: get_RAG_pipeline \t Status: Completed")
  return rag_pipeline


def format_rewrite_answer(question, rewritten_query):
    try:
        json_obj = json.loads(rewritten_query)
        standalone_question = json_obj["standalone_question"]
        better_search_query = json_obj["better_search_query"]
        return standalone_question, better_search_query
    except Exception as E:
        print(f"Error occured while formatting the LLM query rewriting answer.\n Query: {question}\n LLM Answer: {rewritten_query}\n Exception: {E}")


def inference(query_rewriter, rag_pipeline, handler):
    print("Inside Q&A function")
    while True:
        free_gpu_memory()
        question = input("\n Enter a query. Enter 'exit' to quit. ")
        if question == "exit":
            break
        rewritten_query = query_rewriter.rewrite_query(query=question)
        standalone_question, better_search_query = format_rewrite_answer(question, rewritten_query)
        complete_query = standalone_question +" (OR) "+ better_search_query
        print("Rewritten Query:\t",complete_query)
        answer = rag_pipeline(complete_query, callbacks=[handler])['result']
        print("Answer:\t",answer)


def main():
    '''Start up function to load documents, create splits, generate embeddings, create retriever, load model, and create RAG pipeline'''

    print("Function: main \t Status: Started")
    free_gpu_memory()
    handler = StdOutCallbackHandler()

    #Load retriever
    retriever = ingest.get_retriever()

    #Load LLM
    model = get_model()
    
    #Load LLM transformer pipeline
    llm_pipeline = get_llm_pipeline(model)

    #Get conversation memory
    memory = get_memory()

    #Load Query Rewriter chain
    query_rewrite_chain = get_query_rewrite_chain(llm_pipeline, memory)

    #Get Query Rewriter object
    query_rewriter = Query_Rewriter(llm_pipeline=llm_pipeline, memory=memory, query_rewrite_chain=query_rewrite_chain)

    #Get RAG prompt
    prompt_template = get_prompt()

    #Get the RAG pipeline
    rag_pipeline = get_RAG_pipeline(llm_pipeline, retriever, prompt_template, memory)
    
    #Inference
    inference(query_rewriter, rag_pipeline, handler)

    print("Function: main \t Status: Completed")


if __name__ == "__main__":
    main()








