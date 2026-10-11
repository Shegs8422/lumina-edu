# Attribution

## Base model

Lumina-Edu is a derivative work of **`NCAIR1/N-ATLaS`**, the multilingual
Nigerian large language model fine-tuned from Llama-3 8B.

- Maintainer: National Centre of Artificial Intelligence and Robotics (NCAIR)
- Powered by: **Awarri Technologies**
- Initiative of: **The Federal Ministry of Communications, Innovation and
  Digital Economy**
- Licence: [N-ATLaS Terms of Use](https://huggingface.co/NCAIR1/N-ATLaS)
- Paper: *N-ATLaS-LLM: A Multilingual African Language Model* (2025)

## Required attribution string

Per Section 2 of the N-ATLaS Terms of Use, this string must appear wherever
Lumina-Edu or its outputs are presented:

> N-ATLaS is an initiative of the Federal Ministry of Communications,
> Innovation and Digital Economy, and powered by Awarri Technologies.

## Renaming requirement

Because Lumina-Edu renames N-ATLaS, Section 2 of the Terms requires the model
to carry the suffix **"Powered by Awarri."** Accordingly:

- Model / adapter name: **`lumina-edu-natlas-adapter-powered-by-awarri`**
- Product name: **Lumina-Edu — Powered by Awarri**

The suffix appears in the model card, the FastAPI `/` response, the frontend
footer and every benchmark artefact.

## Dataset attribution

Training and evaluation data was harvested from the **University of Ibadan
Institutional Repository** via OAI-PMH, and is used for open academic research
under the repository's own terms. Source content remains the property of the
University of Ibadan and the individual authors. See `DATASET_CARD.md` for the
full provenance trail; every record retains its OAI handle.

## Licence conditions we are bound by

| Condition | How we comply |
|---|---|
| Same-licence distribution | Derivative released under the N-ATLaS Terms |
| "Powered by Awarri" suffix | Carried in the model name and UI |
| Required attribution | Stated above and in every output |
| ≤ 1,000 active end-users (30-day rolling) | Demo deployment is a NAIC submission artefact, not a public service |
| No commercial use without a separate agreement | Lumina-Edu is submitted as a competition entry; no commercial deployment |
| Transparent reporting of accuracy and limitations | `DATASET_CARD.md` §7 and the benchmark report state limitations explicitly |

## Citation

```bibtex
@misc{awagptv1_2025,
  title     = {N-ATLaS-LLM: A Multilingual African Language Model},
  author    = {Awarri Technologies and National Information Technology
               and Development Agency},
  year      = {2025},
  publisher = {Hugging Face}
}
```