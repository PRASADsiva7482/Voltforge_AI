# Verified synthetic data v1

This directory contains the VFAI-009 training-data release. It is generated only by VoltForge-owned grammars and deterministic verifiers; no external AI model, hosted evaluator, web result, private project, or legacy unverified corpus contributes training text.

The pipeline is fail-closed:

- exact board, pin, component, and wiring labels must resolve through the checksum-bound electronics corpus;
- circuit labels must match the complete terminal-to-pin recipe;
- every retained firmware source has a source-bound receipt from Arduino CLI 1.5.1 and Arduino AVR Boards 1.8.6;
- compiler-repair examples retain both the expected failing compile and successful corrected compile receipts;
- duplicates and held-out release-evaluation collisions are rejected before sharding;
- all structured actions remain proposal-only and require user confirmation;
- every shard is a native task-record v1 JSONL shard with an approved governance manifest.

`python tools/build_verified_synthetic_data.py --check` performs a no-write verification and does not require a local compiler. `--recompile` additionally checks the retained receipts against a fresh local pinned-toolchain run. `--write` is the only normal release-writing path and recompiles every firmware case before inclusion.

The first release compiles firmware for the exact Arduino UNO R3, Mega 2560 Rev3, and Nano classic targets. The other curated boards remain fully represented in board, pin, wiring, circuit, structured-output, uncertainty, and safety tasks; their firmware is deliberately rejected until exact cores are separately pinned and verified.
