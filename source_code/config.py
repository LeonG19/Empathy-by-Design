from yacs.config import CfgNode as CN

def get_dataset_config(dataset_name: str):
    cfg = CN()
    cfg.NAME = dataset_name
    cfg.DATA_DIR = f"data/QA_healthcare"
    cfg.SPLIT = "test.csv"           # file with {"question","answer"} per line
    cfg.TEXT_KEY = "question"
    cfg.REF_KEY = "answer"
    cfg.CHUNK_SIZE = 400
    cfg.CHUNK_OVERLAP = 60
    return cfg

def get_llm_config(backend: str, use_finetuned: bool):
    cfg = CN()
    cfg.BACKEND = backend              # "openai"|"hf"|"vllm"|"ollama"
    
    if backend == "hf":
        cfg.PROVIDER = "hf"
        cfg.MODEL = "meta-llama/Llama-3.1-8B-Instruct"
        if use_finetuned:
            cfg.FINETUNED_PATH = "/mnt/lagarza3/llm-healthcare/source_code/train/llama-3.1-8b-ft"
            cfg.LORA = "train/llama_3.1_8B_dpo"
            cfg.USE_FINETUNED = True
        else:
            cfg.USE_FINETUNED = False
        cfg.API_BASE = ""
        
    else:
        cfg.PROVIDER = "openai" if backend == "openai" else "hf"
        cfg.MODEL = "gpt-4o-mini" if backend == "openai" else "meta-llama/Llama-3.1-8B-Instruct"
        cfg.API_BASE = ""
        cfg.FINETUNED_PATH = ""
    cfg.EXTRA = CN(new_allowed=True)
    cfg.TEMPERATURE = 0.3
    cfg.MAX_TOKENS = 150
    
    return cfg

def get_retriever_config(name: str):
    cfg = CN()
    cfg.NAME = name                    # "rankify" | "faiss"
    cfg.TOP_K = 50
    cfg.METHOD = "bm25"                # for rankify retriever
    cfg.INDEX_PATH = f"indexes/{name}_default"   # directory (FAISS) or path prefix (Rankify)
    cfg.CORPUS_DIR = "data/vector_db_data"                
    cfg.BUILD_IF_MISSING = True
    cfg.QUERY_ENCODER = "text-embedding-3-large"
    cfg.EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   
    return cfg


def get_reranker_config(name: str):
    cfg = CN()
    cfg.NAME = name                    # "none"|"rankify"
    cfg.METHOD = "RankLLM"             # RankLLM|MonoT5|CrossEncoder|ColBERT
    cfg.PARAMS = CN(new_allowed=True)
    cfg.PARAMS.MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    cfg.MAX_PASSAGES = 20
    return cfg

def get_rag_config(recipe: str, use_finetuned):
    cfg = CN()
    cfg.RECIPE = recipe                # "vanilla"|"fid"|"hyde"
    cfg.CONTEXT_K = 3
    if recipe == "none":
        print("using base prompt")
        '''
        cfg.SYSTEM_PROMPT = (
    "You are a careful, evidence-based clinical assistant specialized in healthcare, "
    "geriatrics, dementia, and Alzheimer's disease. Start your answer with a brief, "
    "compassionate one-line sentence acknowledging the user's concern, then provide a "
    "clear, simple, and friendly explanation. Do not invent facts. Be polite and kind.")

        '''
        cfg.SYSTEM_PROMPT=(
            "You are a careful, evidence-based clinical assistant specialized in healthcare"
        )
        
        
        
        cfg.USER_TEMPLATE = "Question: {question}\nAnswer:"

    else:
        print("using rag prompt")
        cfg.SYSTEM_PROMPT = (
    "You are a careful, evidence-based clinical assistant specialized in healthcare, "
    "geriatrics, dementia, and Alzheimer's disease. Start your answer with a brief, "
    "compassionate one-line sentence acknowledging the user's concern, then provide a "
    "clear, simple, and friendly explanation. Do not invent facts. Be polite and kind."
        )
        cfg.USER_TEMPLATE = (
            "Use the following context to answer the 'Question'':\n"
            "{context}\n"
            "Question: {question}\nAnswer:"
        )
    return cfg

def get_eval_config():
    cfg = CN()
    # Add the new metric keys to the list if you want them computed:
    # "st_sem"      -> Sentence-Transformers cosine (HF)
    # "embed_cos"   -> Embedding cosine (OpenAI or HF)
    cfg.METRICS = [ "GEval"] #["nli", "rougeL", "bleu", "bertscore","st_sem", "embed_cos", "GEval", "g_empathy",  "rougeL", "bleu", "bertscore","st_sem", "embed_cos", "GEval", "nli", "g_empathy" ] #"rougeL", "bleu", "bertscore",

    cfg.SAVE_PREDICTIONS = True
    cfg.SAVE_TRACES = True

    # Sentence-Transformers semantic similarity
    cfg.ST = CN()
    cfg.ST.MODEL = "sentence-transformers/all-mpnet-base-v2"
    cfg.ST.THRESHOLD = 0.10          # optional: collect pairs below this

    # Embedding cosine similarity (OpenAI or HF)
    cfg.EMBEDSIM = CN()
    cfg.EMBEDSIM.BACKEND = "openai"  # "openai" | "hf"
    cfg.EMBEDSIM.MODEL = "text-embedding-3-large"   # or e.g. "sentence-transformers/all-MiniLM-L6-v2" if BACKEND="hf"
    cfg.EMBEDSIM.API_BASE = ""       # for self-hosted OpenAI-compatible servers (optional)
    cfg.EMBEDSIM.THRESHOLD = 0.10    # optional: collect pairs below this
    return cfg


def get_experiment_cfg():
    cfg = CN()
    cfg.SEED = 42
    cfg.OUT_DIR = "results"
    cfg.NAME = "rag_experiment"
    return cfg



def get_config(dataset: str, llm_backend: str, retriever: str, reranker: str, recipe: str, use_finetuned: bool):
    cfg = CN()
    cfg.EXPERIMENT = get_experiment_cfg()
    cfg.DATASET = get_dataset_config(dataset)
    cfg.LLM = get_llm_config(llm_backend, use_finetuned)
    cfg.RETRIEVER = get_retriever_config(retriever)
    cfg.RERANKER = get_reranker_config(reranker)
    cfg.RAG = get_rag_config(recipe, use_finetuned)
    cfg.EVAL = get_eval_config()
    return cfg
