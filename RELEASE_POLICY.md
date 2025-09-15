# EvoLib Release Policy

EvoLib is in active development. To keep versions transparent and reproducible, we follow a simple release policy.

## Versioning
- EvoLib uses [PEP 440](https://peps.python.org/pep-0440/) compliant versions:
  - **Beta versions**: `0.x.ybN`
  - **Release candidates**: `0.x.yrcN`
  - **Stable versions**: `0.x.y`

## PyPI and Git Tags
- **Every version** (beta, RC, stable) is published to PyPI and tagged in Git.  
- This ensures reproducibility for users and automated tools.

## GitHub Releases
- **GitHub Releases** are created **only** when changes are relevant to users:
  - Breaking changes in API or configuration.
  - New examples or major features.
  - Important bug fixes affecting real usage.
- **Minor/internal changes** (documentation cleanups, refactorings, typos) are published to PyPI and tagged, but **do not get a GitHub Release**.

## Pre-release Flag
- GitHub Releases for **betas** (`bN`) and **RCs** (`rcN`) are marked as *Pre-release*.  
- Stable versions (`0.x.y`) are published as normal releases.

