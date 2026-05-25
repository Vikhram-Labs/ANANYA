# Research Directions — ANANYA

## Publishable Title (working)

**AnanyaBench: Evaluating Multilingual Constitutional Reasoning in Parameter-Efficient Indian Legal Small Language Models**

## Paper Structure

1. **Abstract** — sovereign legal SLM gap; low-resource Indic; T4-feasible QLoRA
2. **Introduction** — digital India, access to justice, hallucination risk in legal AI
3. **Related Work** — Legal-BERT variants, IndicLLM, constitutional NLP, RAG in law
4. **Corpus** — constitution + amendments + SC summaries; translation methodology
5. **Method** — CPT → SFT; glossary-constrained MT; RAG citation format
6. **AnanyaBench** — 7 task families, metrics, human eval protocol
7. **Experiments** — base model ablations (1.5B vs 3B), CPT on/off, RAG on/off
8. **Results** — per-language tables, retrieval recall, hallucination rate
9. **Ethics** — not legal advice; government text licensing; bias audit
10. **Conclusion** — open weights, reproducible Colab pipeline

## Ablation Studies

| Ablation | Hypothesis |
|----------|------------|
| No CPT | SFT-only worse on rare articles |
| No glossary MT | Terminology drift in hi/ta |
| 3B vs 1.5B | Diminishing returns on T4 |
| RAG vs parametric | RAG reduces hallucination_rate |
| Language-specific adapters | MoE or per-lang LoRA vs single adapter |

## Future Work

- Constitutional graph RAG (articles → cross-refs)
- Human-in-the-loop advocate review set
- Fine-tuned multilingual embeddings (IndicBERT-v2 legal)
- Federated fine-tune across state legal aid clinics (privacy-preserving)
- Alignment with official Hindi Rajbhasha glossary updates

## Multilingual Legal AI Opportunities

- Cross-lingual **article entailment** (Art. 14 ↔ state equivalents)
- **Amendment tracking** temporal QA
- **Scheme eligibility** reasoning (governance + constitution intersection)
- Low-resource languages beyond priority five (as, or, kn, ml, pa)
