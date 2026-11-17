"""Synthetic ground-truth validation for IsoFinal.

Estimates the THX10 degradation (axial blur, Z sampling, noise) from the
anisotropic data, builds isotropic phantoms with known truth, degrades them the
same way, and scores trained models against that truth. Self-contained: nothing
outside this folder is modified. See synthetic/README.md.
"""
