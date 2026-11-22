# AI Disclosure
- **AI assistant used**: Claude (Anthropic). It helped design the pipeline, write code (`code/vsum.py`, `tools/validate.py`, `run.sh`) and draft this documentation. All code was reviewed and run by the team.
- **Pretrained model used at inference**: OpenCLIP ViT-B-32 (LAION-2B, `laion2b_s34b_b79k`), zero-shot only, with no fine-tuning.
- **No generative AI output is used as data.** No ground-truth annotations from EPIC-KITCHENS-100 validation or FPVSum are used.
