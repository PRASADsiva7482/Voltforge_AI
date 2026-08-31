import json
from pathlib import Path
import socket
import tempfile
import threading
import time
import unittest
from unittest import mock

from model.gen1.config import Gen1Config
from model.gen1.model import VoltForgeGen1
from model.gen1.optimization import (
    InferenceOptimizationProfile,
    OptimizationProfileError,
    REFERENCE_PROFILE,
    VFAI017_SAFE_PROFILE,
)
from model.gen1.runtime import (
    CancellationToken,
    GenerationCancelled,
    GenerationDeadlineExceeded,
    GenerationOptions,
    LocalGen1Runtime,
    LocalRuntimeError,
    RuntimeState,
    select_local_device,
)
from model.registry_manager import (
    RuntimeContract,
    canonical_json_bytes,
    initialize_signing_key,
    json_file_bytes,
    sha256_bytes,
    sha256_file,
    sign_document,
)
import model.gen1.runtime as runtime_module
from model.tokenizer import BASE_VOCAB_SIZE, VoltForgeTokenizer


class TestLocalGen1Runtime(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.trust_store = self.root / "trust" / "trusted-keys.json"
        self.private_key = self.root / "keys" / "private.pem"
        self.key_id = "vf-runtime-test-key-v1"
        initialize_signing_key(
            self.private_key,
            self.trust_store,
            key_id=self.key_id,
        )

    def tearDown(self):
        self.temporary.cleanup()

    def _artifact(
        self,
        artifact_id: str = "vfdlm-g1-edge-v1.0.0-test",
        *,
        release_status: str = "approved",
        activation_eligible: bool = True,
        offline_experimental: bool = False,
    ) -> Path:
        artifact = self.root / artifact_id
        model_root = artifact / "model"
        tokenizer_root = artifact / "tokenizer"
        model_root.mkdir(parents=True)
        tokenizer_root.mkdir(parents=True)

        tokenizer = VoltForgeTokenizer(vocab_size=BASE_VOCAB_SIZE)
        tokenizer.save(
            tokenizer_root,
            release_status="approved",
            lineage={
                "sourceIds": ["vf-src-runtime-test-v1"],
                "shardIds": ["vf-shard-runtime-test-v1"],
            },
        )
        config = Gen1Config(
            vocab_size=BASE_VOCAB_SIZE,
            max_sequence_length=24,
            d_model=16,
            n_layers=1,
            n_heads=2,
            n_kv_heads=1,
            d_ff=32,
            dropout=0.0,
            seed=17017,
        )
        model = VoltForgeGen1.for_training(config, device="cpu")
        model.save_checkpoint(model_root)

        compatibility = RuntimeContract.current().as_compatibility()
        (artifact / "compatibility.json").write_bytes(json_file_bytes(compatibility))
        policy = {
            "schemaVersion": 1,
            "artifactId": artifact_id,
            "releaseStatus": release_status,
            "servingEnabled": release_status == "approved" and activation_eligible,
            "activationEligible": activation_eligible,
            "offlineExperimentalInferenceEnabled": offline_experimental,
            "maximumInputTokens": 16,
            "maximumNewTokens": 8,
            "samplingEnabled": True,
            "streamingEnabled": True,
            "promptLoggingAllowed": False,
        }
        (artifact / "generation-policy.json").write_bytes(json_file_bytes(policy))
        (artifact / "provenance.json").write_bytes(
            json_file_bytes({"schemaVersion": 1, "source": "runtime-test"})
        )

        roles = {
            "model/config.json": "model-config",
            "model/weights.pt": "model-weights",
            "model/manifest.json": "checkpoint-model-manifest",
            "tokenizer/vocab.json": "tokenizer-vocabulary",
            "tokenizer/merges.json": "tokenizer-merges",
            "tokenizer/tokenizer_config.json": "tokenizer-config",
            "tokenizer/tokenizer_manifest.json": "tokenizer-manifest",
            "compatibility.json": "runtime-compatibility",
            "generation-policy.json": "generation-policy",
            "provenance.json": "package-provenance",
        }
        files = []
        for path in sorted(item for item in artifact.rglob("*") if item.is_file()):
            relative = path.relative_to(artifact).as_posix()
            files.append(
                {
                    "path": relative,
                    "role": roles[relative],
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        manifest = {
            "schemaVersion": 2,
            "artifactKind": "vfdlm-gen1-local-model-package",
            "artifactId": artifact_id,
            "immutable": True,
            "runtime": "pytorch-gen1-v1",
            "compatibilityPath": "compatibility.json",
            "generationPolicyPath": "generation-policy.json",
            "provenancePath": "provenance.json",
            "packageSha256": sha256_bytes(canonical_json_bytes(files)),
            "model": {
                "architectureId": "vfdlm-gen1-decoder-v1",
                "checkpointFormat": "pytorch-weights-only-state-dict-v1",
                "parameterCount": config.parameter_count(),
                "contextLength": config.max_sequence_length,
                "vocabSize": config.vocab_size,
                "quantization": "fp32",
                "configPath": "model/config.json",
                "weightsPath": "model/weights.pt",
                "checkpointManifestPath": "model/manifest.json",
            },
            "tokenizer": {
                "tokenizerId": "vfdlm-byte-bpe",
                "version": "1.0.0",
                "contractVersion": "1.0.0",
                "releaseStatus": "approved",
                "vocabSize": BASE_VOCAB_SIZE,
                "manifestPath": "tokenizer/tokenizer_manifest.json",
            },
            "releaseState": {
                "releaseStatus": release_status,
                "activationEligible": activation_eligible,
            },
            "files": files,
        }
        signed = sign_document(
            manifest,
            digest_field="manifestSha256",
            private_key_path=self.private_key,
            key_id=self.key_id,
        )
        (artifact / "artifact-manifest.json").write_bytes(json_file_bytes(signed))
        return artifact

    def _runtime(self) -> LocalGen1Runtime:
        return LocalGen1Runtime(
            trust_store_path=self.trust_store,
            requested_device="cpu",
        )

    def test_approved_artifact_loads_once_and_reports_ready(self):
        artifact = self._artifact()
        runtime = self._runtime()

        runtime.load(artifact)
        runtime.load(artifact)
        health = runtime.health()

        self.assertEqual(runtime.state, RuntimeState.READY)
        self.assertTrue(health["ready"])
        self.assertTrue(health["runtimeOperational"])
        self.assertEqual(health["artifactId"], artifact.name)
        self.assertEqual(health["parameterCount"], 6_704)
        self.assertEqual(health["contextLength"], 24)
        self.assertEqual(health["device"], "cpu")
        self.assertEqual(health["loadCount"], 1)
        self.assertFalse(health["generationNetworkAccess"])
        self.assertFalse(health["trainingAvailableAtRuntime"])
        self.assertFalse(health["downloadAvailableAtRuntime"])

    def test_process_runtime_rejects_a_second_model_instance(self):
        first = self._artifact("vfdlm-g1-edge-v1.0.1-test")
        second = self._artifact("vfdlm-g1-edge-v1.0.2-test")
        runtime = self._runtime()
        runtime.load(first)

        with self.assertRaises(LocalRuntimeError) as raised:
            runtime.load(second)

        self.assertEqual(raised.exception.code, "MODEL_RUNTIME_ALREADY_LOADED")
        self.assertEqual(runtime.artifact_id, first.name)

    def test_health_exposes_loading_state_during_materialization(self):
        artifact = self._artifact("vfdlm-g1-edge-v1.0.3-test")
        runtime = self._runtime()
        entered = threading.Event()
        release = threading.Event()
        original_verify = runtime_module.verify_artifact_directory
        failures = []

        def delayed_verify(*args, **kwargs):
            entered.set()
            if not release.wait(timeout=5):
                raise AssertionError("test did not release delayed artifact verification")
            return original_verify(*args, **kwargs)

        def load():
            try:
                runtime.load(artifact)
            except Exception as error:  # pragma: no cover - asserted below
                failures.append(error)

        with mock.patch.object(
            runtime_module,
            "verify_artifact_directory",
            side_effect=delayed_verify,
        ):
            worker = threading.Thread(target=load)
            worker.start()
            self.assertTrue(entered.wait(timeout=5))
            health = runtime.health()
            self.assertEqual(health["state"], "loading")
            self.assertEqual(health["code"], "MODEL_RUNTIME_LOADING")
            self.assertFalse(health["ready"])
            with self.assertRaises(LocalRuntimeError) as raised:
                runtime.unload()
            self.assertEqual(raised.exception.code, "MODEL_RUNTIME_LOADING")
            release.set()
            worker.join(timeout=10)

        self.assertFalse(worker.is_alive())
        self.assertEqual(failures, [])
        self.assertEqual(runtime.state, RuntimeState.READY)

    def test_greedy_generation_streaming_and_restart_are_reproducible(self):
        artifact = self._artifact("vfdlm-g1-edge-v1.1.0-test")
        runtime = self._runtime()
        options = GenerationOptions(max_new_tokens=5, mode="greedy")
        runtime.load(artifact)

        first = runtime.generate("LED safety", options=options)
        streamed = list(runtime.stream_generate("LED safety", options=options))
        runtime.unload()
        self.assertEqual(runtime.state, RuntimeState.UNAVAILABLE)
        runtime.load(artifact)
        restarted = runtime.generate("LED safety", options=options)

        self.assertEqual(first.token_ids, restarted.token_ids)
        self.assertEqual(first.text, restarted.text)
        self.assertEqual("".join(item.delta for item in streamed), first.text)
        self.assertTrue(streamed[-1].done)
        self.assertEqual(streamed[-1].finish_reason, first.finish_reason)
        self.assertLessEqual(first.completion_tokens, 5)
        self.assertGreater(first.prompt_tokens, 0)
        self.assertEqual(runtime.health()["loadCount"], 2)

    def test_bounded_sampling_is_seed_reproducible(self):
        artifact = self._artifact("vfdlm-g1-edge-v1.2.0-test")
        runtime = self._runtime()
        runtime.load(artifact)
        options = GenerationOptions(
            max_new_tokens=6,
            mode="sample",
            temperature=0.7,
            top_k=12,
            top_p=0.8,
            seed=442,
        )

        first = runtime.generate("GPIO", options=options)
        second = runtime.generate("GPIO", options=options)

        self.assertEqual(first.token_ids, second.token_ids)
        self.assertEqual(first.text, second.text)
        self.assertLessEqual(first.completion_tokens, 6)

    def test_kv_cache_matches_unoptimized_reference_and_threads_restore(self):
        artifact = self._artifact(
            "vfdlm-g1-edge-v0.2.2-test",
            release_status="experimental",
            activation_eligible=False,
            offline_experimental=True,
        )
        original_threads = runtime_module.torch.get_num_threads()
        reference = LocalGen1Runtime(
            trust_store_path=self.trust_store,
            requested_device="cpu",
            optimization_profile=REFERENCE_PROFILE,
        )
        optimized = LocalGen1Runtime(
            trust_store_path=self.trust_store,
            requested_device="cpu",
            optimization_profile=VFAI017_SAFE_PROFILE,
        )
        options = GenerationOptions(max_new_tokens=7, mode="greedy")

        reference.load(artifact, allow_experimental=True)
        expected = reference.generate("GPIO timing", options=options)
        reference.unload()
        optimized.load(artifact, allow_experimental=True)
        actual = optimized.generate("GPIO timing", options=options)
        optimized.unload()

        self.assertEqual(actual.token_ids, expected.token_ids)
        self.assertEqual(actual.text, expected.text)
        self.assertEqual(runtime_module.torch.get_num_threads(), original_threads)

    def test_unavailable_optimization_falls_back_to_safe_profile(self):
        artifact = self._artifact(
            "vfdlm-g1-edge-v0.2.3-test",
            release_status="experimental",
            activation_eligible=False,
            offline_experimental=True,
        )
        selected = InferenceOptimizationProfile(
            profile_id="vfai018-test-selected",
            kv_cache_enabled=True,
            attention_backend="sdpa",
            torch_threads=2,
            mmap_weights=False,
            quantization="fp32",
            dynamic_batch_max_size=1,
        )
        runtime = LocalGen1Runtime(
            trust_store_path=self.trust_store,
            requested_device="cpu",
            optimization_profile=selected,
        )
        original_apply = runtime_module.apply_model_optimizations
        calls = 0

        def fail_once(model, profile, device):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise OptimizationProfileError(
                    "MODEL_OPTIMIZATION_UNAVAILABLE", "Simulated unavailable kernel."
                )
            return original_apply(model, profile, device)

        with mock.patch.object(
            runtime_module, "apply_model_optimizations", side_effect=fail_once
        ):
            runtime.load(artifact, allow_experimental=True)

        health = runtime.health()
        self.assertEqual(health["optimizationProfile"], VFAI017_SAFE_PROFILE.as_dict())
        self.assertEqual(
            health["optimizationFallbackCode"], "MODEL_OPTIMIZATION_UNAVAILABLE"
        )
        self.assertEqual(calls, 2)
        runtime.unload()

    def test_approved_runtime_rejects_unsigned_profile_override(self):
        artifact = self._artifact("vfdlm-g1-edge-v1.2.1-test")
        runtime = LocalGen1Runtime(
            trust_store_path=self.trust_store,
            requested_device="cpu",
            optimization_profile=REFERENCE_PROFILE,
        )

        with self.assertRaises(LocalRuntimeError) as raised:
            runtime.load(artifact)

        self.assertEqual(raised.exception.code, "MODEL_OPTIMIZATION_OVERRIDE_DENIED")

    def test_policy_and_context_budgets_are_enforced(self):
        artifact = self._artifact("vfdlm-g1-edge-v1.3.0-test")
        runtime = self._runtime()
        runtime.load(artifact)

        bounded = runtime.generate(
            "ok",
            options=GenerationOptions(max_new_tokens=100),
        )
        self.assertLessEqual(bounded.completion_tokens, 8)

        with self.assertRaises(LocalRuntimeError) as raised:
            list(runtime.stream_generate("x" * 40))
        self.assertEqual(raised.exception.code, "MODEL_INPUT_TOKEN_BUDGET_EXCEEDED")

    def test_cancellation_and_deadline_stop_without_degrading_runtime(self):
        artifact = self._artifact("vfdlm-g1-edge-v1.4.0-test")
        runtime = self._runtime()
        runtime.load(artifact)
        cancellation = CancellationToken()
        cancellation.cancel()

        with self.assertRaises(GenerationCancelled) as cancelled:
            list(runtime.stream_generate("LED", cancellation=cancellation))
        self.assertEqual(cancelled.exception.generated_tokens, 0)

        with self.assertRaises(GenerationDeadlineExceeded) as deadline:
            list(runtime.stream_generate("LED", deadline_monotonic=time.monotonic() - 1))
        self.assertEqual(deadline.exception.generated_tokens, 0)
        self.assertEqual(runtime.state, RuntimeState.READY)
        self.assertEqual(runtime.health()["activeRequests"], 0)

    def test_experimental_load_is_explicit_and_policy_bound(self):
        denied = self._artifact(
            "vfdlm-g1-edge-v0.2.0-test",
            release_status="experimental",
            activation_eligible=False,
            offline_experimental=False,
        )
        runtime = self._runtime()

        with self.assertRaises(LocalRuntimeError) as raised:
            runtime.load(denied, allow_experimental=True)

        self.assertEqual(raised.exception.code, "MODEL_EXPERIMENTAL_INFERENCE_DISABLED")
        self.assertEqual(runtime.state, RuntimeState.FAILED)

        permitted = self._artifact(
            "vfdlm-g1-edge-v0.2.1-test",
            release_status="experimental",
            activation_eligible=False,
            offline_experimental=True,
        )
        runtime.load(permitted, allow_experimental=True)
        health = runtime.health()
        self.assertEqual(health["state"], "degraded")
        self.assertFalse(health["ready"])
        self.assertTrue(health["runtimeOperational"])
        self.assertTrue(health["experimentalOffline"])

    def test_generation_requires_valid_options_and_local_device(self):
        artifact = self._artifact("vfdlm-g1-edge-v1.5.0-test")
        runtime = self._runtime()
        runtime.load(artifact)

        with self.assertRaises(LocalRuntimeError) as raised:
            list(runtime.stream_generate("LED", options=GenerationOptions(max_new_tokens=0)))
        self.assertEqual(raised.exception.code, "MODEL_GENERATION_OPTIONS_INVALID")
        self.assertEqual(select_local_device("cpu").type, "cpu")
        with self.assertRaises(LocalRuntimeError) as raised:
            select_local_device("remote")
        self.assertEqual(raised.exception.code, "MODEL_DEVICE_INVALID")

    def test_checked_in_experimental_runtime_loads_and_generates_offline(self):
        project_root = Path(__file__).resolve().parents[1]
        artifact = (
            project_root
            / "model"
            / "registry"
            / "artifacts"
            / "vfdlm-g1-edge-v0.1.1-runtime"
        )
        trust_store = (
            project_root / "model" / "registry" / "trust" / "trusted-keys.json"
        )
        if not artifact.is_dir():
            self.skipTest("ignored local VFAI-017 runtime package is unavailable")
        runtime = LocalGen1Runtime(
            trust_store_path=trust_store,
            requested_device="cpu",
        )

        with mock.patch.object(
            socket.socket,
            "connect",
            side_effect=AssertionError("runtime attempted network access"),
        ):
            runtime.load(artifact, allow_experimental=True)
            result = runtime.generate(
                "LED resistor safety",
                options=GenerationOptions(max_new_tokens=2, mode="greedy"),
            )
        self.assertLessEqual(result.completion_tokens, 2)
        self.assertEqual(runtime.health()["state"], "degraded")
        runtime.unload()


if __name__ == "__main__":
    unittest.main()
