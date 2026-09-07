# Vendored: MiniMax H3 prompt-writing skill

`SKILL.md` and `references/*.txt` in this directory are copied verbatim from
MiniMax's own repository:

- Upstream: https://github.com/MiniMax-AI/MiniMax-H3/tree/main/skills/h3-prompt-writing
- Licence: MiniMax H3 Community License Agreement
  (https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)

They are the authoring guide `api_wrapper/prompt_rewriter.py` feeds to the MiMo
model as its system prompt, so the rewriter produces prompts in exactly the
structure H3 was trained on. Keeping them on disk (rather than fetching at
request time) means the rewrite path has no network dependency beyond the model
call itself, and pins the guide to a known revision.

SHA-256 of the copies, so a drift from upstream is detectable:

    SKILL.md               a7000443588ca3f145e3b3fd8900f14e0325dc460bd811268fac89a9dc8e56d0
    references/base-en.txt 2cfebc096a6e08370f288d468d90b60f7f9bcb938f94bf090816e910e48e75fc
    references/ref-en.txt  1e574f356716ad55612247ffb7bbccbcdb484ad96599d63c7dca1af186b1fab7

To refresh, re-copy from upstream and update the hashes above. To point the
rewriter at a different copy without touching this directory, set
`H3_PROMPT_SKILL_DIR` to a directory with the same layout.
