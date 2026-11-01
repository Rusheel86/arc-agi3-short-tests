# K8V4 KV-cache port (Apache-2.0)

`e083_k8v4_port.py` stores the model's attention memory (the KV cache) with 8-bit (fp8) keys and 4-bit (nvfp4) values,
so the same GPU memory holds about 1.27x more tokens (ledger E083: +0.0016 nats per token). E098 and E154 spend that
memory on depth: 19 games at once with a 160,768-token window.

**Licence: Apache License 2.0, not MIT-0.** The file contains andreasgru's Apache-2.0 code (the nvfp4 helpers and the
value quantiser of `kvq-patch.py`, Copyright 2026 Andreas Gruhler; `LICENSE` here is that project's licence file), and
its own header puts the whole file under Apache-2.0. The changes we made are listed in its header and in `../NOTICE`.

**It is the exact file the notebook runs.** The notebook carries it zlib + base64 encoded in its E073 prepare cell (`E098_PORT`),
pinned by sha256 and size, and writes it before the server starts. This copy is byte-identical:
sha256 `f5c1cb2bc9a5fca03d1c2136b7386db7f1f6a1a4c9e798df1b5a9e881c66675d`, 83,910 bytes (Windows line endings, as
pinned). It changes nothing unless the server runs with `--kv-cache-dtype fp8` and `ARC3_KVQ_K8V4=1`; everything it
inserts into vLLM carries the marker `ARC3-KVQ-E083`. The names in its header that look like paths (`vendor/kv-quant`,
`experiments/e062_kvq_port.py`) are files of the project repository, not of this release.
