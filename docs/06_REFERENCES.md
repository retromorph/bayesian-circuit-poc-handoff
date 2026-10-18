# 06 — Literature, repositories and implementation compatibility

The coding agent should consult these primary resources, pin accessed Git commits, and document if implementation choices differ from the cited method. A reference alone is not a reproduction.

## Closest method competitors

1. **Bair, Xu, Sun & Kolter (ICML 2026), _Compressed Sensing for Capability Localization in Large Language Models_.** Paper: https://proceedings.mlr.press/v306/bair26a.html ; code: https://github.com/locuslab/llm-components . Important **necessary knockout head sets** baseline. Their capability localization objective is not automatically the same as **sufficient retained circuits**. Optional comparable diagnostic if readily executable.
2. **Butler et al. (NeurIPS 2025), _ProxySPEX: Inference-Efficient Interpretability via Sparse Feature Interactions in LLMs_.** Paper: https://proceedings.neurips.cc/paper_files/paper/2025/hash/68dced1638d71d4d21598f79ae91262e-Abstract-Conference.html ; implementation `shapiq.ProxySPEX`: https://shapiq.readthedocs.io/en/latest/auto_examples/approximators/surrogate_and_sparse/plot_proxyspex.html ; repo: https://github.com/mmschlk/shapiq . The exposed interaction values are not necessarily direct held-out response predictions. For fair *prediction* comparison, use the analogous GBDT surrogate separately.
3. **MIB: A Mechanistic Interpretability Benchmark (2025).** Landing page: https://github.com/aaronmueller/MIB ; circuit localization implementation: https://github.com/hannamw/MIB-circuit-track . Source supplies IOI tasks, GPT-2 Small, EAP-IG and circuit evaluation conventions. Default benchmark edge-patching and our in-pool head-zero-ablation are **different** metrics; label any comparisons carefully.
4. **InterpBench: semi-synthetic transformers with known circuits (NeurIPS 2024).** Repo: https://github.com/FlyingPumba/InterpBench . Useful optional validation beyond the hand-built XOR model. Repository uses Poetry and may have older dependency constraints; isolate or wrap it rather than polluting this uv environment.
5. **TransformerLens.** Repo: https://github.com/TransformerLensOrg/TransformerLens ; hooks: https://github.com/TransformerLensOrg/TransformerLens/blob/main/docs/source/content/hook_system.md . `run_with_hooks` / hook tensors can intervene on GPT-2 heads. Test the actual `hook_z` tensor layout and release data and gradients before scaling.

## Methodological context

6. **Geiger et al. (JMLR 2025), _Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability_.** https://jmlr.org/papers/v26/23-0058.html . Important for why an attribution/regression explanation is not automatically a causally faithful high-level abstraction.
7. **Wang et al. (2023), _Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 Small_.** https://arxiv.org/abs/2211.00593 . Canonical IOI task and component-level intervention benchmark.
8. **Conmy et al. (2023), _Towards Automated Circuit Discovery for Mechanistic Interpretability_.** https://arxiv.org/abs/2304.14997 . ACDC and automated pruning background.
9. **Mueller et al. (ICML 2025), _MIB: A Mechanistic Interpretability Benchmark_.** https://proceedings.mlr.press/v267/mueller25a.html . Related evaluation rationale and known pitfalls.
10. **Asiaee (UAI 2026), _Certified Interventional Fidelity_.** https://proceedings.mlr.press/v337/asiaee26d.html . Important comparison for statistical validity claims; Bayesian credible intervals are not equivalent to anytime-valid confidence guarantees.

## Environment documentation

- **uv**: https://docs.astral.sh/uv/ and PyTorch guide https://docs.astral.sh/uv/guides/integration/pytorch/ . Pin the wheel index compatible with actual remote NVIDIA driver, not a guessed CUDA version.
- **MLflow server**: https://mlflow.org/docs/latest/tracking/server/ and CLI https://mlflow.org/docs/latest/api_reference/cli.html . MLflow 3.5+ has host/CORS security options; keep this pilot on loopback behind SSH.
- **NVIDIA RTX A4000**: https://www.nvidia.com/en-eu/products/workstations/rtx-a4000/ . 16 GB GDDR6; microbatch Gate C and avoid retaining huge activation caches.

## Repository policy

- External code must be referenced by specific commit SHA in the final report and run manifest, with local license file preserved when vendoring anything.
- Do not invent equivalence to an official method if only a simplified independent implementation was executed; label it `replica`, `adaptation`, or `proxy` as appropriate.
- Prefer 1–2 carefully integrated dependencies over importing every research framework. An A4000 POC should be reproducible and debuggable.
