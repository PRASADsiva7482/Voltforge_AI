# Verified synthetic data v1.2

This directory contains the VFAI-009 training-data release. It is generated only by VoltForge-owned grammars and deterministic verifiers; no external AI model, hosted evaluator, web result, private project, or legacy unverified corpus contributes training text.

The pipeline is fail-closed:

- exact board, pin, component, and wiring labels must resolve through the checksum-bound electronics corpus;
- circuit labels must match the complete terminal-to-pin recipe;
- every retained firmware source has a source-bound receipt from the exact FQBN and profile in `toolchains.v1.json`;
- compiler-repair examples retain both the expected failing compile and successful corrected compile receipts;
- duplicates and held-out release-evaluation collisions are rejected before sharding;
- all structured actions remain proposal-only and require user confirmation;
- every shard is a native task-record v1 JSONL shard with an approved governance manifest.

`python tools/build_verified_synthetic_data.py --check` performs a no-write verification and does not require a local compiler. `--recompile` additionally checks the retained receipts against a fresh local pinned-toolchain run. `--write` recompiles every firmware case before inclusion. `--refresh` may rewrite reports, shards, and governance manifests without compilation only when every retained receipt remains source-, FQBN-, command-, and toolchain-bound to the current grammar and lock. Compilation uses a local Arduino CLI config, a workspace-temporary build path, and an offline-after-install policy. One isolated build state is reused per exact FQBN during a release operation, and `--jobs 0` lets the pinned CLI parallelize compilation across local CPU cores before the temporary session is removed.

The v1.2 release compiles the exact Arduino UNO R3, Mega 2560 Rev3, Nano classic, Nano Every, UNO R4 WiFi, Leonardo, Micro, ESP32-DevKitC V4/WROOM-32E-N4, ESP32-S3-DevKitC-1/WROOM-1-N8, Raspberry Pi Pico, and Raspberry Pi Pico 2 targets. The Pico 2 profile uses the pinned Arduino-Pico source checkout and recorded submodule revisions because the official Arduino Mbed RP2040 package does not expose a Pico 2 FQBN.
