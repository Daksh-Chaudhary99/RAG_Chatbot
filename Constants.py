import gc
import torch
import transformers
from torch import cuda, bfloat16
from GPUtil import showUtilization as gpu_usage

hf_auth = ''

source_location = 'test_documents'

model_id = 'meta-llama/Llama-2-7b-chat-hf'

model_path = 'model'

bge_persist_directory = 'embeddings/bge'

contriever_persist_directory = 'embeddings/contriever'

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)

def free_gpu_memory():
    '''Free up GPU memory'''

    print("GPU usage before releasing memory")
    gpu_usage()
    torch.cuda.empty_cache()

    gc.collect()

    print("GPU Usage after releasing memory")
    gpu_usage()