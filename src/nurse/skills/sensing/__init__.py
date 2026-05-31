"""Sensing skills — real modality-specialized model calls (vision, audio scene).

Each is a background `Skill`: it reads an input (frame grab / audio clip), calls a model
chosen for that modality, and returns a timestamped `SkillFinding` the Front Voice speaks.
Disabled gracefully when its model isn't installed (sensing is enrichment, never required).
"""
