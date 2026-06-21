"""
SearchQAEnvServer
"""

from typing import Optional
import threading
import json
import os
import re
import argparse
import datasets

from .utils import Config
from .retriever import get_retriever
from .reward_score import compute_score_em, compute_score_em_format

file_path = os.path.dirname(os.path.abspath(__file__))

import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class NotInitializedError(Exception):
    pass


TEST_ITEM_RANGE = {
    "nq": (0, 3610),
    "triviaqa": (3610, 14923),
    "popqa": (14923, 29190),
    "hotpotqa": (29190, 36595),
    "2wikimultihopqa": (36595, 49171),
    "musique": (49171, 51588),
    "bamboogle": (51588, 51713),
}

TRAIN_ITEM_RANGE = {
    "nq": (51713, 130881),
    "hotpotqa": (130881, 221328),
}

ITEM_RANGE = {
    "test": (0, 51713),
    "train": (51713, 221328),
}

faiss_gpu = os.environ.get("SEARCHQA_FAISS_GPU", "False").lower() == "true"
retrieval_method = os.environ.get("SEARCHQA_RETRIEVAL_METHOD", "e5")
retrieval_topk = int(os.environ.get("SEARCHQA_RETRIEVAL_TOPK", "3"))
index_path = os.environ.get(
    "SEARCHQA_INDEX_PATH",
    os.path.join(file_path, "..", "retrieve_data", "e5_Flat.index"),
)
corpus_path = os.environ.get(
    "SEARCHQA_CORPUS_PATH",
    os.path.join(file_path, "..", "retrieve_data", "wiki-18.jsonl"),
)
retrieval_model_path = os.environ.get(
    "SEARCHQA_RETRIEVAL_MODEL_PATH",
    os.path.join(file_path, "..", "retrieve_data", "e5-base-v2"),
)
retrieval_use_fp16 = (
    os.environ.get("SEARCHQA_RETRIEVAL_USE_FP16", "True").lower() == "true"
)
retrieval_batch_size = int(os.environ.get("SEARCHQA_RETRIEVAL_BATCH_SIZE", "512"))

# New: maximum consecutive identical search queries; 0 means disabled (default)
max_consecutive_same_query = int(
    os.environ.get("SEARCHQA_MAX_CONSECUTIVE_SAME_QUERY", "0")
)


class SearchQAEnvServer:
    """
    SearchQAEnvServerEnvServer
    """

    def __init__(self) -> None:

        config = Config(
            retrieval_method=retrieval_method,  # or "dense"
            index_path=index_path,
            corpus_path=corpus_path,
            retrieval_topk=retrieval_topk,
            faiss_gpu=faiss_gpu,
            retrieval_model_path=retrieval_model_path,
            retrieval_pooling_method="mean",
            retrieval_query_max_length=256,
            retrieval_use_fp16=retrieval_use_fp16,
            retrieval_batch_size=retrieval_batch_size,
        )

        self.retriever = get_retriever(config)
        self._max_id = 0
        self.env = {}
        self.ls = []
        self._lock = threading.Lock()
        self.is_format_valid = {}

        # New: per-env tracking for consecutive identical search queries
        self._last_search_query = {}   # env_idx -> str or None
        self._same_query_streak = {}   # env_idx -> int

        train_dataset = datasets.load_dataset(
            "parquet",
            data_files=os.path.join(file_path, "queries", "train.parquet"),
            keep_in_memory=False,
        )["train"]
        test_dataset = datasets.load_dataset(
            "parquet",
            data_files=os.path.join(file_path, "queries", "test.parquet"),
            keep_in_memory=False,
        )["train"]

        self.dataset = {
            "test": test_dataset,
            "train": train_dataset,
        }

    def create(self, item_id: int = 0) -> int:
        with self._lock:
            env_idx = self._max_id
            self._max_id += 1

        self.env[env_idx] = self._fetch_data(
            item_id
        )  # redundancy fetch to prevent NoneType Error
        self.ls.append(env_idx)

        # init format & streak states
        self.is_format_valid[env_idx] = True
        self._last_search_query[env_idx] = None
        self._same_query_streak[env_idx] = 0

        return env_idx

    def step(self, env_idx, response: str):
        """
        Perform a step in the environment with the given action.
        Input:
            env_idx: the index of the environment
            action: a string in the format "<search> query </search>" or "<answer> answer </answer>"
        Output:
            observation: the observation after taking the action
            reward: the reward received after taking the action
            done: whether the episode is done
            info: additional information (not used here)
        Raises:
            ValueError: if the action is not a valid string format
        """
        self._check_env_idx(env_idx)
        reward = 0
        done = False
        observation = ""

        strict_pattern = (
            r"^\ufeff?[ \t\r\n]*"                       # leading whitespace (0+), allow optional BOM
            r"<think>.*?</think>"                       # exactly one think block
            r"[ \t\r\n]*"                               # whitespace between think and action (0+)
            r"<(?P<action>search|answer)>"              # exactly one action block
            r"(?P<content>.*?)"
            r"</(?P=action)>"
            r"[ \t\r\n]*\Z"                             # trailing whitespace (0+)
        )

        if isinstance(response, str):  # for llm output
            pattern = r"<(search|answer)>(.*?)</\1>"
            match = re.search(pattern, response, re.DOTALL)
            if match:
                content = match.group(2).strip()
                action = match.group(1)
            else:
                content = ""
                action = None

            if not re.fullmatch(strict_pattern, response, flags=re.DOTALL):
                self.is_format_valid[env_idx] = False
        else:
            raise ValueError(f"Invalid action type: {type(action)}")

        reward = 0 if self.is_format_valid[env_idx] else 0 

        if action == "search":
            search_query = content
            logger.info(f"Search query: {search_query}")

            # New: consecutive identical query guard (disabled if max_consecutive_same_query <= 0)
            if max_consecutive_same_query > 0:
                last_q = self._last_search_query.get(env_idx, None)
                if last_q is not None and search_query == last_q:
                    self._same_query_streak[env_idx] = self._same_query_streak.get(env_idx, 0) + 1
                else:
                    self._same_query_streak[env_idx] = 1
                    self._last_search_query[env_idx] = search_query

                if self._same_query_streak[env_idx] >= max_consecutive_same_query:
                    done = True
                    observation = (
                        f"Failed: repeated the same search query for "
                        f"{self._same_query_streak[env_idx]} consecutive times "
                        f"(threshold={max_consecutive_same_query})."
                    )
                    return observation, reward, done, None

            search_results = self._search(search_query)
            observation = f"<information>{search_results.pop(0).strip()}</information>"

        elif action == "answer":
            # Check if the answer is correct
            format_score = compute_score_em_format(
                response,
                self.env[env_idx]["reward_model"]["ground_truth"],
            )
            score = compute_score_em(
                solution_str=response,
                ground_truth=self.env[env_idx]["reward_model"]["ground_truth"],
                format_score=format_score,
            )

            if score == 1:
                done = True
                reward += 1 
                observation = (
                    "Congratulations! You have answered the question correctly!!!"
                )
            else:
                done = True
                observation = (
                    "Sorry, your answer is incorrect. Do not stop early. Avoid any inference, "
                    "assumptions, or guesses anything that is not explicitly supported by the "
                    "retrieved sources!"
                )
        else:
            observation = (
                "Your previous action is invalid. If you want to search, you should put the "
                "query between <search> and </search>. If you want to give the final answer, "
                "you should put the answer between <answer> and </answer>. Please try again."
            )

        return observation, reward, done, None

    def observation(self, env_idx):
        self._check_env_idx(env_idx)
        question = self.env[env_idx]["question"]
        # user_prompt = f"""You must first reason inside <think>...</think> before you take any action. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search>, and it will return the top searched results between <information> and </information>. You can search as many times until you have gathered sufficient, verifiable information to answer accurately. You can search as many times as you want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer> without detailed illustrations. For example, <answer> xxx </answer>. \n\n Question: {question.strip()}"""

        user_prompt = f"""You must first reason inside <think>...</think> before you take any action. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search>, and it will return the top searched results between <information> and </information>. You can search as many times until you have gathered sufficient, verifiable information to answer accurately. Do not stop early. Avoid any inference, assumptions, or guesses anything that is not explicitly supported by the retrieved sources. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer> without detailed illustrations. For example, <answer> xxx </answer>. \n\n Question: {question.strip()}"""
        return user_prompt

    def reset(self, env_idx, item_id: Optional[int] = None):
        self._check_env_idx(env_idx)
        self.env[env_idx] = self._fetch_data(item_id)
        self.is_format_valid[env_idx] = True

        # New: reset streak states
        self._last_search_query[env_idx] = None
        self._same_query_streak[env_idx] = 0

    def _search(self, search_query: str):
        results, scores = self.retriever.search(
            query=[search_query], num=3, return_score=True
        )
        # Format response
        resp = []
        combined = []
        for doc, score in zip(results, scores):
            combined.append({"document": doc, "score": score})
        resp.append(combined)
        result = [self._passages2string(r) for r in resp]
        logger.info(f"Search results: {result}\nRAW: {resp}")
        return result

    def _passages2string(self, retrieval_result):
        format_reference = ""
        for idx, doc_item in enumerate(retrieval_result):
            content = doc_item["document"]["contents"]
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
        return format_reference

    def _check_env_idx(self, env_idx):
        if env_idx not in self.env:
            raise IndexError(f"Env {env_idx} not found")
        if self.env[env_idx] is None:
            raise NotInitializedError(f"Env {env_idx} not initialized")

    def _fetch_data(self, item_id: int):
        """
        Fetch data from the dataset based on the item_id.
        """
        _id = None
        for mode, r in ITEM_RANGE.items():
            if r[0] <= item_id < r[1]:
                _id = item_id - r[0]
                return self.dataset[mode][_id]
        if _id is None:
            raise ValueError(f"Item id {item_id} is out of range.")

    def __del__(self):
        for idx in self.ls:
            del self.env[idx]
            print(f"-------Env {idx} closed--------")

    def close(self, id):
        try:
            self._check_env_idx(id)
            self.ls.remove(id)
            del self.env[id]

            # New: cleanup streak states
            if id in self._last_search_query:
                del self._last_search_query[id]
            if id in self._same_query_streak:
                del self._same_query_streak[id]
            if id in self.is_format_valid:
                del self.is_format_valid[id]

            print(f"-------Env {id} closed--------")
            return True
        except Exception as e:
            print(f"Error closing env {id}: {e}")
            return False


searchqa_env_server = SearchQAEnvServer()

