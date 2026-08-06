# Recommended GitHub repository settings

These settings require repository-owner access and must be applied manually by Hitesh Kumar. They do
not change package metadata, tags, releases, or the frozen v0.1.0 result.

## About panel

Use this exact description:

> Reproducible multi-agent RL research platform for cooperative exploration in procedural environments.

Leave the **Website** field empty until a real documentation or demo URL exists. Enable **Releases**
and **Issues**. Discussions are optional; enable them only if there is capacity to moderate research
questions. Leave Packages unused unless a future release actually publishes a package.

Recommended topics:

```text
multi-agent-reinforcement-learning
reinforcement-learning
marl
mappo
ppo
pettingzoo
pytorch
embodied-ai
procedural-generation
reproducible-research
```

Manual steps:

1. Open the repository on GitHub.
2. In the right-hand **About** panel, select the gear icon.
3. Paste the description exactly as written above.
4. Leave **Website** blank.
5. Add the ten topics, preserving their spelling.
6. Select **Releases** and **Issues**; select **Discussions** only if desired.
7. Leave **Packages** unselected unless a real package is published later.
8. Save changes.

## Social preview

Upload [`docs/assets/kovara9-social-preview.png`](assets/kovara9-social-preview.png). It is an original
1280×640 project graphic with a subtle agent/grid motif and no metrics or third-party marks.

Manual steps:

1. Open **Settings → General** for the repository.
2. Find **Social preview**.
3. Select **Edit** or **Upload an image**.
4. Upload `docs/assets/kovara9-social-preview.png` from the checkout.
5. Confirm the crop keeps the KOVARA-9 title and visual fully visible.

Regenerate the source file with `python scripts/generate_readme_assets.py`. Provenance and the exact
hash are recorded in [`docs/assets/readme-assets-manifest.json`](assets/readme-assets-manifest.json).

## Features and merge hygiene

Recommended owner review after the documentation pull request:

- keep Issues enabled so the repository templates are usable;
- keep the existing v0.1.0 release published;
- do not create a v0.2.0 tag until release scope is reviewed;
- consider branch protection for `main` with the existing CI workflow required; and
- avoid enabling Wikis because the maintained documentation already lives under `docs/`.

This document is advisory only. No GitHub setting is changed by repository code.
