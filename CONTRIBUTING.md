# Contributing to Awesome-Deblurring-Resources

Thank you for helping keep this collection accurate and useful for researchers and practitioners working on image and video deblurring.

## What Belongs in This Repository

Good additions include work where deblurring, blur formation/synthesis, or reconstruction from blurred observations is a central research problem. Relevant areas include image and video deblurring, motion and defocus blur, blind deconvolution, event/gyro/spike-assisted deblurring, rolling-shutter correction with deblurring, blur-aware NeRF/Gaussian Splatting, and deblurring-specific datasets or benchmarks.

Please avoid adding generic image-restoration papers or general-purpose datasets solely because they can be used for deblurring. The connection to blur/deblurring should be substantive.

## How to Contribute

1. **Fork the repository.**
2. **Clone your fork.**
   ```bash
   git clone https://github.com/<your-username>/Awesome-Deblurring-Resources.git
   ```
3. **Create a branch.**
   ```bash
   git checkout -b add/<paper-or-dataset-name>
   ```
4. **Make your changes** in the appropriate section.
5. **Commit and push.**
   ```bash
   git add .
   git commit -m "Add <paper-or-dataset-name>"
   git push origin add/<paper-or-dataset-name>
   ```
6. **Open a pull request** against the main repository.

## Adding Papers

Use the existing table format:

```markdown
| Venue | Paper | Link |
|-------|-------|------|
| CVPR | [Paper Title](https://official-paper-link) | [Code](https://github.com/owner/repo) |
```

### Paper guidelines

- Prefer the official conference/journal page, proceedings page, DOI, or arXiv record for the paper link.
- Prefer the authors' official implementation for the code/project link.
- If no implementation is publicly available, use `-` rather than an unofficial reimplementation.
- **Use the publication venue and publication year when a paper has been formally published.** For example, a paper posted to arXiv in 2024 and published at WACV 2025 belongs in the 2025 section as WACV.
- Use `arXiv` as the venue only when no formal publication venue is known yet.
- Avoid duplicate entries. Check both the title and paper/arXiv link before adding a new row.
- Keep titles and venue names consistent with the official publication record.

## Adding Datasets

Use the existing table format:

```markdown
| Name | Description | Link |
|------|-------------|------|
| Dataset Name | Concise description of its deblurring use, scale, and distinguishing characteristics. | [Dataset](https://official-link) |
```

### Dataset guidelines

- Prefer the official project, dataset, or authors' repository link.
- Include datasets designed for deblurring, blur synthesis, blur-aware reconstruction, or a clearly related benchmark.
- Keep descriptions concise and factual. Include dataset scale when it is clearly documented by the authors.
- Do not add broad general-purpose datasets unless they contain a dedicated blur/deblurring benchmark or protocol relevant to this collection.

## Fixes and Maintenance

Corrections are welcome, including:

- broken or outdated links;
- incorrect publication years or venues;
- duplicate entries;
- missing official code/project links;
- dataset statistics or descriptions that conflict with the authors' documentation;
- spelling and formatting errors.

When correcting factual information, please include the authoritative source in the pull-request description when practical.

## General Guidelines

- Keep additions focused and high quality rather than maximizing list size.
- Prefer primary sources over aggregator pages.
- Preserve the existing Markdown formatting unless a change is intentionally restructuring the repository.
- Be respectful and constructive in issues, pull requests, and reviews.

Thank you for contributing!
