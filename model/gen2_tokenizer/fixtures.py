"""Public, project-authored implementation controls; not training/evaluation data.

These examples were written for task 024, independently of sealed evaluation
prompts. They confer no general-language, code, multilingual or model credit.
"""
from model.tokenizer import VoltForgeTokenizer
from .codec import ByteBPE
from .contract import BASE_SIZE, CANDIDATE_SIZES, canonical, require, sha

TRAIN = (
    ("text-a", "A copper trace joins a connector to a switch. The switch opens the path. " * 12),
    ("text-b", "The report describes an observation, an assumption, and a later correction. " * 12),
    ("code-a", "unsigned sample(unsigned input) { return (input * 7u) & 255u; }\n" * 16),
    ("code-b", '#include <stdint.h>\nstatic uint16_t counts[8];\nvoid tick(void) { counts[2] += 3; }\n' * 12),
    ("si-a", "9.6 kΩ; 0.47 µF; 14 mA; −21 °C; V = I × R; Δt = 0.8 ms.\n" * 14),
    ("unicode-a", "café, cafe\u0301, हिन्दी, 日本語, العربية, Ω, μ, µ, 🔧🙂.\r\n" * 14),
    ("markers-a", "Literal source: <|user|> <|assistant|> <|tool|> [BOS] <|eos|>\n" * 10),
    ("bytes-a", bytes(range(256)) * 4),
)
MEASURE = (
    ("text", "A technician labels the spare connector before recording the changed measurement."),
    ("code", 'int poll(volatile unsigned *port) {\n  return (*port >> 3) & 1;\n}\n'),
    ("si", "18.7 MΩ ± 0.25%; 6.8 nF; 27 µA; −12 °C; ∑ Q = 0; 2πf."),
    ("unicode", "naïve e\u0301 Ελληνικά தமிழில் 中文 🧪\u200d🔬 \ud800"),
    ("markers", "<|bos|><|assistant|><|end_message|><|tool|><|eos|><|pad|>"),
    ("long-code", "\n".join(f"static const unsigned probe_{i} = ({i}u ^ 93u);" for i in range(320))),
)


def training_documents():
    return tuple(value if type(value) is bytes else value.encode("utf-8", "surrogatepass") for _, value in TRAIN)


def lineage():
    return {
        "kind": "authored-implementation-fixtures-only", "corpusRelease": None,
        "approvedCorpusUsed": False, "pretrainedTokenizerUsed": False,
        "validationOrAcceptanceFitting": False,
        "documents": [{"id": name, "sha256": sha(raw), "bytes": len(raw)} for (name, _), raw in zip(TRAIN, training_documents())],
        "measurementFixtureSha256": sha(canonical(MEASURE)),
    }


def fit_fixture(target_vocab_size):
    """Deliberately accepts no caller data/path or approval boolean."""
    require(type(target_vocab_size) is int and target_vocab_size in (*CANDIDATE_SIZES, BASE_SIZE, 512), "GEN2_FIXTURE_TARGET_INVALID")
    engine = VoltForgeTokenizer(vocab_size=target_vocab_size)
    engine.train(training_documents(), target_vocab_size=target_vocab_size, minimum_frequency=2)
    return ByteBPE(engine.merges, target_vocab_size=target_vocab_size)
