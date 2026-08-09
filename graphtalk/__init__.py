"""Primer generation for GraphQA prompts.

Importable rather than script-local so cluster runs and local spot checks share
one code path. Submodules are imported explicitly (`from graphtalk import
primers`) so that importing the package pulls in nothing heavy.
"""
