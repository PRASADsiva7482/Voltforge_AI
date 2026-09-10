"""Bounded deterministic training of owned BPE merges from exact corpus bytes."""
import time
from model.tokenizer import VoltForgeTokenizer
from model.gen2_tokenizer.codec import ByteBPE
from model.gen2_tokenizer.contract import require
from .corpus import POLICY, fitting_inputs


def fit(target):
    require(type(target) is int and target in POLICY["candidateVocabularyTargets"], "GEN2_FITTING_TARGET_INVALID")
    documents, lineage = fitting_inputs()
    require(len(documents) == lineage["documents"] and sum(map(len, documents)) == lineage["bytes"] <= POLICY["maximumTrainBytes"], "GEN2_FITTING_BYTE_BUDGET")
    started = time.perf_counter()
    engine = VoltForgeTokenizer(vocab_size=target)
    engine.train(documents, target_vocab_size=target, minimum_frequency=POLICY["minimumPairFrequency"])
    tokenizer = ByteBPE(engine.merges, target_vocab_size=target)
    seconds = time.perf_counter()-started
    require(seconds <= POLICY["maximumFitSeconds"], "GEN2_FITTING_TIME_BUDGET")
    require(sum(len(value) for value in tokenizer._values[16:]) <= POLICY["maximumVocabularyBytes"], "GEN2_FITTING_VOCAB_BUDGET")
    return tokenizer, lineage, seconds
