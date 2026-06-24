"""Tokenizer wrappers (visual / action / text).

In stage 1 we adopt the open-source Cosmos-Tokenizer for visual tokenization
and a LLaMA-3 BPE tokenizer for text. Action tokenization is a small learned
codebook per environment, also implemented in stage 1.
"""
