
import argparse, os, json, random
from typing import List, Dict, Any
import numpy as np
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams
from deepeval.test_case import LLMTestCase
from config import get_config
import pandas as pd
from metrics import sts_similarity_hf, embeddings_similarity
import statistics
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch, json, re

# optional deps
def _maybe_import(name):
    try: return __import__(name)
    except Exception: return None

rouge_score = _maybe_import("rouge_score")
sacrebleu = _maybe_import("sacrebleu")
bertscore = _maybe_import("bert_score")
faiss = _maybe_import("faiss")

def parse_args():
    p = argparse.ArgumentParser(description="Unified health-RAG runner")
    p.add_argument("--rag_method", choices=["none", "vanilla","fid","hyde"], required=True)
    p.add_argument("--retriever", choices=["rankify","faiss"], required=False)
    p.add_argument("--retriever_method", default="bm25",
                   help="rankify retriever (bm25|hybrid|colbert|gte|bge...)")
    p.add_argument("--reranker", choices=["none","rankify"], default="none")
    p.add_argument("--reranker_method", default="RankLLM",
                   help="rankify reranker (RankLLM|MonoT5|CrossEncoder|ColBERT)")
    p.add_argument("--llm", choices=["openai","hf"], required=True)
    p.add_argument("--use_finetuned", action="store_true")
    p.add_argument("--dataset", required=True)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--context_k", type=int, default=3)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--metrics_only", action="store_true")
    p.add_argument("--results_dir", type = str, required=False)
    return p.parse_args()

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def ensure_results_dir(args):
    retr = args.retriever if args.retriever else "none"
    ft = ""
    if args.use_finetuned: ft = "ft"
    path = os.path.join("results", ft, args.dataset, args.rag_method,
                        args.llm, f"{retr}_{args.reranker}")
    os.makedirs(path, exist_ok=True)
    return path

def load_test_data(path: str):
    data = pd.read_csv(path)
    
    return data

# ---------- LLM init ----------
def init_llm(cfg):
    provider = cfg.LLM.PROVIDER  # "openai" or "hf"
    # Decide which model path to load
    # If you're storing the merged DPO model in a separate path, set LLM.FINETUNED_PATH in config.py
    # If you instead overwrite LLM.MODEL when use_finetuned=True in get_config(), this fallback will still work.

    model_path = cfg.LLM.MODEL 


    if getattr(cfg.LLM, "USE_FINETUNED", False) and model_path is None:
        raise ValueError(
            "USE_FINETUNED=True but LLM.FINETUNED_PATH is not set in config. "
            "Set cfg.LLM.FINETUNED_PATH to the merged model directory produced by dpo_LLM.py."
        )
    

    api_base = cfg.LLM.API_BASE
    temp = cfg.LLM.TEMPERATURE
    max_tokens = cfg.LLM.MAX_TOKENS

    if provider == "openai":
        from openai import OpenAI
        client = OpenAI(base_url=api_base or None)
        model = model_path  # for openai we still call it 'model'
        def _gen(prompt: str):
            rsp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role":"system","content": cfg.RAG.SYSTEM_PROMPT},
                    {"role":"user","content": prompt}
                ],
                temperature=temp,
                max_tokens=max_tokens,
            )
            return rsp.choices[0].message.content.strip()
        return _gen

    elif provider in {"hf"}:
        # Always load via bitsandbytes (your chosen default)
        from llm.base_LLM import HuggingFaceLLM
        llm = HuggingFaceLLM(
            model_name_or_path=model_path,
            max_new_tokens=max_tokens,
            temperature=temp,
            finetuned = cfg.LLM.USE_FINETUNED,# your earlier default was 8-bit; keep or flip to 4-bit per your wrapper
            load_in_8bit=True
        )
        def _gen(prompt: str):
            return llm.generate(prompt=prompt, system_prompt=cfg.RAG.SYSTEM_PROMPT)
        return _gen

    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

from indexing.index_builder import (
    build_or_load_faiss_index,
    FaissRetriever,
    export_corpus_jsonl_for_rankify,
)

def init_retriever(cfg, args):
    # Always ensure we’ve exported/built from the configured corpus folder when present
    corpus_dir = cfg.RETRIEVER.CORPUS_DIR or os.path.join(cfg.DATASET.DATA_DIR, "corpus")

    if cfg.RETRIEVER.NAME == "rankify":
        # Optional: export a plaintext corpus JSONL so Rankify can build its own index if needed
        # (This doesn't replace Rankify's own build process; it just ensures the corpus lives at a known path.)
        export_path = os.path.join(cfg.RETRIEVER.INDEX_PATH, "corpus.jsonl")
        os.makedirs(cfg.RETRIEVER.INDEX_PATH, exist_ok=True)
        export_corpus_jsonl_for_rankify(corpus_dir, export_path)

        from rankify.retrievers import build_retriever
        return build_retriever(
            name=cfg.RETRIEVER.METHOD,
            index_path=cfg.RETRIEVER.INDEX_PATH,
            build_if_missing=cfg.RETRIEVER.BUILD_IF_MISSING,
            query_encoder=cfg.RETRIEVER.QUERY_ENCODER,
            top_k=cfg.RETRIEVER.TOP_K,
        )

    elif cfg.RETRIEVER.NAME == "faiss":
        index, meta = build_or_load_faiss_index(
            corpus_dir=corpus_dir,
            index_dir=cfg.RETRIEVER.INDEX_PATH,
            embed_model_name=cfg.RETRIEVER.EMBED_MODEL,
            rebuild_if_model_mismatch=True,
        )
        return lambda q: FaissRetriever(index, meta, cfg.RETRIEVER.EMBED_MODEL)(q, top_k=cfg.RETRIEVER.TOP_K)

    else:
        raise ValueError(f"Unknown retriever: {cfg.RETRIEVER.NAME}")

def init_reranker(cfg, args):
    if cfg.RERANKER.NAME == "none":
        return None
    try:
        from rankify.rerankers import build_reranker
    except Exception as e:
        raise RuntimeError("rankify (rerankers) not installed") from e
    return build_reranker(name=cfg.RERANKER.METHOD, **dict(cfg.RERANKER.PARAMS))

# ---------- Pipeline helpers ----------
def retrieve_docs(retriever, query: str):
    if hasattr(retriever, "__call__"):
        return retriever(query)
    return retriever.retrieve(query)

def rerank_docs(reranker, query: str, docs: List[Dict[str, Any]], max_passages: int):
    if reranker is None or not docs: return docs[:max_passages]
    ranked = reranker.rerank(query, docs)
    return ranked[:max_passages]

def format_context(docs: List[Dict[str, Any]]):
    out=[]
    for d in docs:
        if isinstance(d, dict):
            txt = d.get("chunk") or d.get("content") or str(d)
        else:
            txt = str(d)
        out.append(txt)
    return "\n\n".join(out)

def slice_until_kth(
    text: str,
    keyword: str,
    *,
    k: int = 1,
    case_sensitive: bool = False,
    min_tokens_required: int = 10,
) -> str:
    """
    Return all words up to (but not including) the k-th occurrence of `keyword`.
    - If the slice has fewer than `min_tokens_required` words, return the full text.
    - If `keyword` is not found, return the full text.

    Example:
        slice_until_kth("a b c a d", "a", k=1) -> "a b c"
        slice_until_kth("short text", "privacy", k=1) -> "short text"
    """
    if not text or not keyword or k < 1:
        return text

    tokens = text.split()
    if not tokens:
        return text

    def match(tok: str) -> bool:
        return tok if case_sensitive else tok.lower() == keyword.lower()

    # find all indices of the keyword
    idxs = [i for i, tok in enumerate(tokens) if match(tok)]
    if not idxs:
        return text

    # choose k-th occurrence (or last if fewer occurrences exist)
    end_idx = idxs[k - 1] if k <= len(idxs) else idxs[-1]

    # slice up to (but not including) that keyword
    result_tokens = tokens[:end_idx]

    # if slice too short, return full text
    if len(result_tokens) < min_tokens_required:
        return text

    return " ".join(result_tokens)





def run_vanilla_rag(args, cfg, results_dir):
    llm = init_llm(cfg)
    retriever = init_retriever(cfg, args)
    reranker = init_reranker(cfg, args)
    ds = load_test_data(os.path.join(cfg.DATASET.DATA_DIR, cfg.DATASET.SPLIT))

    preds, refs = [], []

    for i, ex in ds.iterrows():
        q = ex[cfg.DATASET.TEXT_KEY]
        docs = retrieve_docs(retriever, q)
        docs = rerank_docs(reranker, q, docs, cfg.RERANKER.MAX_PASSAGES)
        context = format_context(docs[:cfg.RAG.CONTEXT_K])
        if i < 10: 
            print (q, context)
        prompt = cfg.RAG.USER_TEMPLATE.format(question=q, context=context)
        ans = llm(prompt)
        ans = slice_until_kth(ans, "<stop>", k=1)
        preds.append(ans)
        refs.append(ex[cfg.DATASET.REF_KEY])
    return preds, refs, ds["question"].values

def run_llm(args, cfg, results_dir):
    llm = init_llm(cfg)
    preds, refs = [],[]
    ds = load_test_data(os.path.join(cfg.DATASET.DATA_DIR, cfg.DATASET.SPLIT))
    for i, ex in ds.iterrows():
        q = ex[cfg.DATASET.TEXT_KEY]
        if cfg.LLM.USE_FINETUNED == False:
           prompt = q   
        else:  
            prompt = cfg.RAG.USER_TEMPLATE.format(question=q)
        ans = llm(prompt)
        preds.append(ans)
        refs.append(ex[cfg.DATASET.REF_KEY])
    return preds, refs,  ds["question"].values
        



def run_fid_rag(args, cfg, results_dir):
    # placeholder FiD: same as vanilla; replace with real FiD encoder/decoder fusion later
    return run_vanilla_rag(args, cfg, results_dir)

def run_hyde_rag(args, cfg, results_dir):
    llm = init_llm(cfg)
    retriever = init_retriever(cfg, args)
    reranker = init_reranker(cfg, args)
    ds = load_jsonl(os.path.join(cfg.DATASET.DATA_DIR, cfg.DATASET.SPLIT))

    preds, refs = [], []
    for ex in ds:
        q = ex[cfg.DATASET.TEXT_KEY]
        pseudo = llm(f"Write a concise, likely answer (may be imperfect):\n\n{q}")
        docs = retrieve_docs(retriever, f"{q}\n{pseudo}")
        docs = rerank_docs(reranker, q, docs, cfg.RERANKER.MAX_PASSAGES)
        context = format_context(docs[:cfg.RAG.CONTEXT_K])
        prompt = cfg.RAG.USER_TEMPLATE.format(question=q, context=context)
        ans = llm(prompt)
        preds.append(ans)
        refs.append(ex.get(cfg.DATASET.REF_KEY, ""))
    return preds, refs


def factual_consistency_nli(reference, candidate, nli):
    """
    Improved factual consistency using NLI.
    - Candidate is the premise (facts to check)
    - Reference is the hypothesis (truth baseline)
    """
    result = nli({"text": candidate, "text_pair": reference})
    label = result["label"].upper()
    score = result["score"]

    if "ENTAILMENT" in label:
        final_score = score
    elif "NEUTRAL" in label:
        final_score = 0.7 * score  # treat neutral as partially consistent
    else:  # contradiction
        final_score = 1 - score

    return final_score


import json, re

import re
import json

def extract_json(text):
    """
    Extract the last numeric value following the word 'score' in the model output.
    If no numeric score is found, return {'score': None, 'reasoning': <raw text>}.
    """

    # Handle dicts directly (if already parsed)
    if isinstance(text, dict):
        return text

    # Ensure text is a string
    if not isinstance(text, str):
        text = str(text)

    # Find all numeric values following "score"
    matches = re.findall(r"score[^0-9\-]*([-+]?\d*\.?\d+)", text, flags=re.IGNORECASE)

    if matches:
        # Take the last numeric instance found
        last_score = float(matches[-1])
        return {"score": last_score, "reasoning": text.strip()}

    # fallback — no numeric score found
    return {"score": None, "reasoning": text.strip()}


def build_factual_prompt(reference, candidate):
    return f"""
You are a strict factual evaluator. Your job is to critically assess how factually consistent the CANDIDATE text is with the REFERENCE text.

Focus ONLY on factual correctness and completeness.
Ignore tone, politeness, grammar, or stylistic differences.

REFERENCE:
{reference}

CANDIDATE:
{candidate}

Grading Instructions:
- Use a score between 0.0 and 1.0 in increments of 0.1 (e.g., 0.0, 0.1, 0.2, …, 1.0).
- Be conservative. Award high scores (≥0.9) **only if every factual statement in the candidate is accurate and covered by the reference**.
- Deduct points for:
  - Missing important facts
  - Adding unsupported or incorrect claims
  - Overgeneralizing or distorting details
  - Contradicting or changing meaning
- The score should roughly mean:
    - **1.0** → Perfect factual alignment, nothing missing or incorrect.
    - **0.9** → Nearly perfect; only trivial differences.
    - **0.8** → Mostly accurate; 1–2 minor factual omissions or wording changes.
    - **0.7** → Several omissions or vague generalizations.
    - **0.6** → Partially accurate; about half of the facts align.
    - **0.5** → Mixed accuracy; substantial parts missing or unclear.
    - **0.4** → Mostly incomplete or partially incorrect.
    - **0.3** → Significant factual errors or omissions.
    - **0.2** → Very little factual overlap.
    - **0.1** → Barely related or mostly incorrect.
    - **0.0** → Completely unrelated or factually false.

You MUST output your result in the following JSON format:

{{
  "score": <float between 0.0 and 1.0>,
  "reasoning": "<brief, precise justification for the score>"
}}
"""





def get_factual_correctness(reference, candidate, generator, tokenizer):
    prompt = build_factual_prompt(reference, candidate)
    response = generator(
        prompt,
        max_new_tokens=300,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )[0]["generated_text"]

    # print(response)


    return response

import os

os.environ["OPENAI_API_KEY"] = "Provide key to use geval with gpt"

# ---------- Metrics ----------
def compute_metrics(preds: List[str], refs: List[str], which: List[str], cfg_eval, questions) -> Dict[str, Any]:
    res = {}
    # Existing metrics (rouge, bleu, bertscore) are still handled as before
    # (assumes you still import rouge_score/sacrebleu/bertscore above)
    if "rougeL" in which and rouge_score is not None:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        vals = [scorer.score(r, p)["rougeL"].fmeasure for p, r in zip(preds, refs)]
        res["rougeL_f"] = float(np.mean(vals))
    if "bleu" in which and sacrebleu is not None:
        res["bleu"] = float(sacrebleu.corpus_bleu(preds, [refs]).score)
    if "bertscore" in which and bertscore is not None:
        P, R, F1 = bertscore.score(preds, refs, lang="en", rescale_with_baseline=True)
        res["bertscore_f1"] = float(F1.mean().item())

    # NEW: Sentence-Transformers semantic similarity
    if "st_sem" in which:
        st_model = cfg_eval.ST.MODEL
        th = cfg_eval.ST.THRESHOLD if hasattr(cfg_eval.ST, "THRESHOLD") else None
        st_out = sts_similarity_hf(preds, refs, model_name=st_model, show_mismatches_below=th)
        res["st_sem_mean"] = st_out["mean"]
        # If you want to persist per-example:
        # res["st_sem_per_example"] = st_out["per_example"]
        if "below_threshold" in st_out:
            res["st_sem_below_threshold_count"] = len(st_out["below_threshold"])

    # NEW: Embedding cosine similarity (OpenAI or HF)
    if "embed_cos" in which:
        backend = cfg_eval.EMBEDSIM.BACKEND
        model = cfg_eval.EMBEDSIM.MODEL
        api_base = cfg_eval.EMBEDSIM.API_BASE or None
        th = cfg_eval.EMBEDSIM.THRESHOLD if hasattr(cfg_eval.EMBEDSIM, "THRESHOLD") else None
        es_out = embeddings_similarity(preds, refs, backend=backend, model_name=model, api_base=api_base, show_mismatches_below=th)
        res["embed_cos_mean"] = es_out["mean"]
        # If you want to persist per-example:
        # res["embed_cos_per_example"] = es_out["per_example"]
        if "below_threshold" in es_out:
            res["embed_cos_below_threshold_count"] = len(es_out["below_threshold"])

    if "GEval" in which:
        correctness_metric = GEval(
            name="Correctness",
            criteria="Determine whether the actual output is factually correct based on the expected output.",
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT])
        scores = []
        for i in range(len(preds)):
            test_case = LLMTestCase(input= questions[i], actual_output= refs[i], expected_output= preds[i])
            correctness_metric.measure(test_case)
            scores.append(correctness_metric.score)
        res["GEval"] = statistics.mean(scores)

    if "nli" in which:
        nli = pipeline(
            "text-classification",
            model="MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
            device=0,          # still GPU
            torch_dtype=torch.float16  
        )
        final_score = []
        for i in range(len(preds)):
            final_score.append(factual_consistency_nli(refs[i], preds[i], nli))
        res["nli score"] = statistics.mean(final_score)

    if "reading" in which:
        import textstat as ts
        final_score =[]
        for i in range(len(preds)):
            fkgl_score = ts.flesch_kincaid_grade(preds[i])
            final_score.append(fkgl_score)
        res["reading"] = statistics.mean(final_score)



    if "Factual_ModLLM" in which:
        import os
        os.environ["HF_TOKEN"] = "provide key to use hf models"
        model_name = "mistralai/mistral-7b-instruct-v0.2"   # choose any instruct model you like

        tokenizer = AutoTokenizer.from_pretrained(model_name, token=os.environ["HF_TOKEN"])

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            token=os.environ["HF_TOKEN"]


        )
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
        final_score_llm = []
        null_indices = []
        for i in range(len(preds)):
            print(f"{i+1} of {len(preds)}")

            base_result_json = get_factual_correctness(reference= refs[i], candidate= preds[i], generator= generator, tokenizer = tokenizer)
            base_result = extract_json(base_result_json)
            score = base_result.get('score') 
            print(score)
            if score is not None:
                	final_score_llm.append(score)
                    
            else:
            	print("------------------------------")
                print(base_result_json)
            	print(base_result)
            	null_indices.append(i)

        if final_score_llm:
            r = statistics.mean(final_score_llm)
            print("Mean score (ignoring nulls):", r)
        else:
        	print("No valid scores found.")
        if null_indices:
            print(f"Re-running for {len(null_indices)} null entries...")
            for i in null_indices:
                retry_json = get_factual_correctness(refs[i], preds[i])
                retry_result = extract_json(retry_json)
                retry_score = retry_result.get('score')
                if retry_score is not None:
                    final_score_llm.append(retry_score)

	    # Final mean after retries
        if final_score_llm:
            r = statistics.mean(final_score_llm)
            print("Final mean after retries:", r)
        res["ModLLM"] = statistics.mean(final_score_llm)
    
    if "polite_eval" in which:

        politeness_metric = GEval(
        name="Politeness",
        # Use steps only (no `criteria`) so they're the sole source of truth.
        evaluation_steps=[
            # Acknowledgement & warmth
            "Does the actual output open (or early on) with a brief, warm acknowledgment of the user's goal/concern (e.g., 'I'm glad you're looking into...', 'I'm sorry you're dealing with...')?",

            # Respectful, non-judgmental tone
            "Is the tone respectful and free of blame, condescension, sarcasm, or lecturing? Penalize any language that shames or scolds the user.",

            # Courteous markers / softeners
            "Does the response use courteous markers or softeners where appropriate (e.g., 'please', 'thank you', 'you might consider', 'could', 'may') rather than hard imperatives?",

            # Inclusive, person-first language
            "Does the response avoid stereotypes and use inclusive or person-first language (e.g., 'people with...' instead of defining by a condition), and avoid unnecessary gendering or assumptions?",

            # Profanity & hostility check
            "Is there an absence of profanity, insults, all-caps shouting, excessive exclamation, or aggressive punctuation (e.g., '!!!')?",

            # Helpful, supportive framing
            "Are suggestions framed as options (not commands), offering help without overpromising? For refusals or limits, does it politely explain why and suggest safe alternatives?",

            # Clarity without bulldozing
            "Is the response clear and concise without talking over the user (e.g., not overly verbose or dismissive)? Penalize abrupt or dismissive closings.",

            # Medical/health sensitivity (generalizable)
            "If health or sensitive topics arise, does the response stay empathetic, avoid moralizing, and refrain from giving absolute judgments about the user's choices?",

            # Consistency with a polite exemplar
            "Compared with a polite exemplar (neutral, supportive, evidence-aware tone), does the actual output maintain a similarly calm, encouraging style?",
            
            # Final vibe check
            "Overall, would a reasonable reader feel respected, supported, and comfortable after reading the actual output?"
        ],
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT])
        scores = []
        for i in range(len(preds)):
                test_case = LLMTestCase(input= questions[i], actual_output= refs[i], expected_output= preds[i])
                correctness_metric.measure(test_case)
                scores.append(correctness_metric.score)
        res["polite_eval"] = statistics.mean(scores)

    if "g_empathy" in which:
        empathy_metric = GEval(
            name="Empathy",
            evaluation_steps=(
                "An empathetic response should acknowledge the user's perspective or feelings, "
                "Assess how well the actual output demonstrates empathy toward the user's message. "
                "show understanding and emotional awareness, and maintain a warm, respectful, and supportive tone. "
                "It should avoid dismissiveness, judgment, or detachment, and instead convey genuine concern or care "
                "appropriate to the context of the input. The overall impression should be that the responder is attentive, kind, and emotionally in tune with the user."
            ),
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT])
        scores = []
        for i in range(len(preds)):
                test_case = LLMTestCase(input= questions[i], actual_output= refs[i], expected_output= preds[i])
                empathy_metric.measure(test_case)
                scores.append(empathy_metric.score)
        res["g_empathy"] = statistics.mean(scores)




            
        


    return res

def main():
    args = parse_args()
    set_seed(args.random_state)

    cfg = get_config(
        dataset=args.dataset,
        llm_backend=args.llm,
        retriever=args.retriever,
        reranker=args.reranker,
        recipe=args.rag_method,
        use_finetuned=args.use_finetuned,
    )
    cfg.RETRIEVER.TOP_K = args.top_k
    cfg.RAG.CONTEXT_K = args.context_k
    if cfg.RETRIEVER.NAME == "rankify": cfg.RETRIEVER.METHOD = args.retriever_method
    if cfg.RERANKER.NAME == "rankify":  cfg.RERANKER.METHOD  = args.reranker_method

    results_dir = ensure_results_dir(args)

    if args.metrics_only:
        questions = pd.read_csv("data/QA_healthcare/test.csv")["question"].values
        preds = []
        refs = []
        limit = 523 #523 max
        with open(str(args.results_dir), "r") as f:
            for i, line in enumerate(f):
                if i >= limit:
                    break
                obj = json.loads(line)
                preds.append(obj["pred"])
                refs.append(obj["ref"])
                

    
    else:
        if args.rag_method == "vanilla":
            preds, refs, questions = run_vanilla_rag(args, cfg, results_dir)
        elif args.rag_method == "fid":
            preds, refs = run_fid_rag(args, cfg, results_dir)
        elif args.rag_method == "none":
            preds, refs, questions = run_llm(args,  cfg, results_dir)
        else:
            preds, refs = run_hyde_rag(args, cfg, results_dir)

    metrics = compute_metrics(preds, refs, list(cfg.EVAL.METRICS), cfg.EVAL, questions)

    if cfg.EVAL.SAVE_PREDICTIONS:
        with open(os.path.join(results_dir,"predictions.jsonl"),"w",encoding="utf-8") as f:
            for p,r in zip(preds, refs):
                f.write(json.dumps({"pred":p,"ref":r})+"\n")
    with open(os.path.join(results_dir,"metrics.json"),"w",encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
