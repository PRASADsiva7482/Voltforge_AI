"""Independently authored precision controls; never corpus or model training input.

Labels describe constructed relationships, not model quality. Generated variants
share families, and reports count those families separately from case totals.
No protected evaluation prompt is opened while constructing these examples.
"""
from data_governance.splitting.signatures import features, sha, words

SCENES = (
    "Marin placed a blue notebook beside the brass compass before leaving the observatory.",
    "Nell carried a scarlet lantern toward the wooden barrel while the harbor bells rang.",
    "Oren planted purple carrots beyond the garden fence after the bees returned home.",
    "Tavi folded the ancient atlas beneath the library clock during the winter storm.",
    "Sera brushed silver glaze across the clay bowl beside the silent pottery wheel.",
    "Ivo balanced the empty basket against the orchard ladder before picking yellow plums.",
    "Rani wrapped the small violin inside velvet cloth before closing the eastern window.",
    "Luka recorded the cold thermometer reading beside a copper tray inside the snowy cabin.",
)
UNRELATED = "Geologists classify volcanic deposits by mineral composition and cooling rate across sedimentary layers."


def document(identity, text, *, family=None):
    return {
        "id": identity, "recordId": identity, "rawSha256": sha(text), "normalizedSha256": sha(text),
        "features": features([text]),
        "lineage": {"kind": "repository-document", "sourceFamilyId": family or identity, "documentFamilyId": identity, "repositoryId": family or identity},
    }


def guard(identity, text):
    return {"id": identity, "keys": [], "features": features([text]), "kind": "authored-precision-control-only"}


def cases():
    controls = []
    for index, scene in enumerate(SCENES):
        family = f"scene-{index}"
        protected = guard("control-guard-" + family, scene)
        # An alphabetic index lists terms without asserting the scene. All terms
        # are present in the document, exposing the unordered-set false positive.
        glossary = "Index entries in alphabetical order. " + "\n".join(
            f"{word}: catalog label for entry {number}; unrelated inventory metadata."
            for number, word in enumerate(sorted(set(words(scene))))
        )
        variants = (
            ("exact", scene, "protected-positive", "Exact copy of the independently authored protected scene"),
            ("normalized", "  " + scene.upper().replace(" ", "\n") + "  ", "protected-positive", "Case/whitespace normalization preserves the scene"),
            ("embedded", UNRELATED + "\n" + scene + "\n" + UNRELATED, "protected-positive", "Exact scene inserted into a larger document"),
            ("glossary", glossary, "unrelated-negative", "Shared vocabulary in an index does not assert or reproduce the protected narrative"),
            ("wide-glossary", ("\n" + UNRELATED * 12 + "\n").join(glossary.splitlines()), "unrelated-negative", "Words are distributed across unrelated index entries and prose"),
            ("unrelated", UNRELATED, "unrelated-negative", "Distinct topic and no shared narrative"),
        )
        for kind, text, label, rationale in variants:
            identity = family + "-" + kind
            controls.append({"id": identity, "family": family, "label": label, "rationale": rationale,
                             "document": document("control-document-" + identity, text), "guard": protected})
    # Code/family/ID signals remain mandatory even if semantic intent is unknown.
    code = "int sample(int input) { int count = input + 7; if (count > 12) return count; return 0; }"
    for name, changed in (
        ("renamed-code", code.replace("sample", "measure").replace("input", "voltage").replace("count", "result")),
        ("parameterized-code", code.replace("sample", "measure").replace("7", "19").replace("12", "31")),
    ):
        controls.append({"id": name, "family": "code-family", "label": "protected-positive", "rationale": "Known shared code family must stay excluded",
                         "document": document("control-document-" + name, changed), "guard": guard("control-guard-code", code)})
    for key in ("id", "recordId", "rawSha256", "normalizedSha256"):
        row, protected = document("identity-document-" + key, UNRELATED), guard("identity-guard-" + key, "A completely different account of distant stars and telescopes.")
        protected[key] = row[key]
        controls.append({"id": "identity-" + key, "family": "identity-family", "label": "protected-positive", "rationale": "Identity equality is an unconditional exclusion",
                         "document": row, "guard": protected})
    for number, (scene, paraphrase) in enumerate((
        (SCENES[0], "Before leaving the observatory, Marin placed the notebook, which was blue, beside a compass made of brass."),
        (SCENES[5], "Before picking plums that were yellow, Ivo balanced an empty basket against the ladder in the orchard."),
    )):
        controls.append({"id": f"paraphrase-{number}", "family": f"scene-{0 if number == 0 else 5}", "label": "protected-positive",
                         "rationale": "Same authored event expressed with reordered clauses; lexical uncertainty must still block admission",
                         "document": document(f"control-paraphrase-{number}", paraphrase), "guard": guard(f"control-paraphrase-guard-{number}", scene)})
    return controls
