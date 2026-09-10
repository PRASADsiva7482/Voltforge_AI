"""Measure exact indexed recall, nonempty controls and bounded-scale resource use."""
import argparse
import ctypes
import json
from pathlib import Path
import sys
import time
import tracemalloc
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_governance.pretraining import indexed, corpus
from data_governance.splitting.partition import plan_partitions
from data_governance.splitting.signatures import sha, features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    # Unit fixture module has no acquired content and imports no evaluation
    # prompts. Its use here is explicit regression evidence, never training data.
    import importlib.util
    spec = importlib.util.spec_from_file_location("pretraining_controls", corpus.AI / "tests/test_pretraining_corpus.py")
    fixtures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixtures)
    rows, guards = fixtures.recall_fixture()
    expected = plan_partitions(rows, guards)
    observed = indexed.scan(rows, guards)
    for key in ("protectedMatches", "duplicateMatches"):
        if observed[key] != expected[key]: raise ValueError("Indexed comparator misses exhaustive matches")
    controls = [fixtures.doc(f"independent-fixture-{i}", " ".join("word" + sha(f"{i}:{j}")[:16] for j in range(12))) for i in range(100)]
    decisions, checked = indexed.partition(controls, [])
    if {row["split"] for row in decisions} != {"train", "validation", "test"}: raise ValueError("Nonempty partition control missing")
    measurements = []
    for size in (1000, 6000):
        # Independent fixture families deliberately avoid ubiquitous shared
        # words. A separate pathological-common-word test checks fail-closed
        # behavior; this benchmark does not imply worst-case subquadratic work.
        sample = [fixtures.doc(f"scale-{i}", " ".join("w" + sha(f"scale:{i}:{j}")[:20] for j in range(12))) for i in range(size)]
        tracemalloc.start()
        started = time.perf_counter()
        result = indexed.scan(sample, guards)
        elapsed = time.perf_counter() - started
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        measurements.append({"documents": size, "guards": len(guards), "seconds": elapsed, "documentsPerSecond": size / elapsed,
                             "peakTracedScanBytes": peak, "measurementScope": "index construction and scan allocations; excludes already-constructed input feature objects", "verifiedPairs": result["verifiedPairs"], "exhaustivePairUpperBound": result["exhaustivePairUpperBound"]})
    # Measure parity against real source/guard inputs independently from the
    # source-controlled corpus build; no protected text is written to the report.
    real_rows, _, _, _ = corpus.load_inputs()
    real_guards, _ = corpus.splitting.protected_registry()
    descriptions = [corpus.describe(row) for row in real_rows]
    full = plan_partitions(descriptions, real_guards)
    actual = indexed.scan(descriptions, real_guards)
    if any(full[key] != actual[key] for key in ("protectedMatches", "duplicateMatches")): raise ValueError("Real-source index recall differs from exhaustive")
    report = {"status": "passed", "fixtureDocuments": len(rows), "fixtureGuards": len(guards), "matchedProtectedFixtures": len(observed["protectedMatches"]),
              "matchedDuplicateEdges": len(observed["duplicateMatches"]), "fixtureRecall": 1.0, "realInputDocuments": len(real_rows), "realProtectedGuards": len(real_guards), "realInputRecall": 1.0,
              "nonemptyControlSplitCounts": {split: sum(row["split"] == split for row in decisions) for split in ("train", "validation", "test")},
              "nonemptyControlAudit": checked["audit"], "measurements": measurements, "semanticCompletenessClaimed": False,
              "largeCorpusScaleAccepted": False, "maximumComparedFixtureDocuments": 6000, "trainingAllowed": False,
              "sourceFingerprints": corpus.fingerprints() + [{"path": "tests/test_pretraining_corpus.py", "sha256": sha((corpus.AI / "tests/test_pretraining_corpus.py").read_bytes())}, {"path": "tools/evaluate_pretraining_index.py", "sha256": sha(Path(__file__).read_bytes())}]}
    corpus.write_immutable(args.report, corpus.data(report))
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
