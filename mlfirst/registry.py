"""Lazy lesson registry: listing lessons does not import NumPy."""
from importlib import import_module

TITLES = {1: 'Learning problems and evidence',
 2: 'Data and experiments',
 3: 'The mathematical language of models',
 4: 'Probability and statistical inference',
 5: 'Optimization and generalization',
 6: 'Measuring what matters',
 7: 'Linear and nonlinear regression',
 8: 'Probability for classification',
 9: 'Neighbors and separating surfaces',
 10: 'Decision trees',
 11: 'Ensembles',
 12: 'Clustering',
 13: 'Low dimensional structure',
 14: 'Association rules',
 15: 'Probabilistic graphical models',
 16: 'Neural networks and backpropagation',
 17: 'Learning from images',
 18: 'Recurrent models for sequences',
 19: 'Attention and transformers',
 20: 'Learning on graphs',
 21: 'Autoencoders and latent variables',
 22: 'Self supervised representation learning',
 23: 'Learning with scarce labels and transferring knowledge',
 24: 'Language models',
 25: 'Generative adversarial models',
 26: 'Diffusion and score based generation',
 27: 'Multimodal learning',
 28: 'Sequential decisions',
 29: 'Learning values',
 30: 'Learning policies',
 31: 'Operating reliable learning systems',
 32: 'An end to end investigation'}


def run_lesson(number, seed=42):
    if number not in TITLES:
        raise ValueError("lesson must be between 1 and 32")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    module = "foundations" if number <= 6 else "classical" if number <= 15 else "neural" if number <= 27 else "decisions"
    function = getattr(import_module(f"mlfirst.{module}"), f"lesson_{number:02d}")
    return function(seed=seed)
