# STAT 236 project showcase website

**Project:** Network-Aware Testing Allocation for Infectious Disease Control Under Limited Resources: The Role of Community Structure and Incomplete Contact Information

**Authors:** Yinpeng Cai, Aniket Jain, Annie Liang, Zidan Luo — Fall 2026

## Preview

Open `index.html` in a browser. The site uses only local HTML, CSS, JavaScript, SVG, PNG, and PDF assets and runs **without a build step, server, network, or third-party packages**. The interactive chart uses actual aggregate rows from `data/optimized_sbm_summary.csv`, embedded into `assets/data.js` so it also works when opened as a local file.

## Publish on GitHub Pages (recommended)

1. Create a new **public** GitHub repository, for example `stat236-testing-showcase`.
2. Upload the **contents of this folder** into the repository root. `index.html` must be at the root.
3. Go to **Settings → Pages → Build and deployment**. Set **Source: Deploy from a branch**, **Branch: main**, **Folder: / (root)**, and save.
4. When GitHub Pages finishes deploying, the site will be available at `https://YOUR-USERNAME.github.io/stat236-testing-showcase/`. Share this URL with the course staff.

Alternatively, upload the **entire website folder contents** to a static-hosting provider such as Netlify, Cloudflare Pages, or an existing university webspace.

## Contents

- `index.html` — web page
- `assets/style.css` — responsive design
- `assets/app.js` — working interactive chart and schematic network
- `assets/data.js` — numerical data for the SBM chart, derived from project summary CSV
- `assets/*.png` — original figures from the group's LaTeX project
- `data/*.csv` — two original summary tables, downloadable from the site
- `report.pdf` — compiled from the supplied `report.tex` and bibliography; 19 pages

## Academic accuracy

The contact network shown in the hero is a **schematic illustration**, not the observed Haslemere network, and its nodes/edges are generated solely for the visual. The interactive **result explorer uses real simulation summary outputs** from the included ZIP (30 paired simulated epidemics for each SBM scenario). The dynamic Haslemere results are drawn from the `extended_policy_table.tex` and `extended_results_text.tex` in the supplied report. The site explicitly notes that only three Haslemere days are measured, that later days are resampled, and that all infections are simulated. No claims of statistical superiority are made where paired Monte Carlo intervals include zero.

## Before submitting

- Replace or supplement PDF if the group updates the written report after this version.
- Re-check the authors and numbers against the final submitted report.
- The source document mentions `https://github.com/annieliang111/STAT236Project`, but a public request returned 404 on 2026-11-05. The site therefore does **not** link to this repository until it is confirmed publicly accessible. You may add a code link to `index.html` once the repository becomes public.
- The site is for an academic demonstration and does not predict public-health outcomes.
