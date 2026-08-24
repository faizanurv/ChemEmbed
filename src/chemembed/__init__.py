"""ChemEmbed — metabolite identification from MS/MS spectra via molecular embeddings.

A convolutional network predicts a 300-dimensional mol2vec embedding from a
spectrum, which is then matched by cosine similarity against a reference
database of pre-computed embeddings.

The trained models and the reference database are distributed separately from
this package because of their size; see the README for download instructions.
"""

__version__ = "1.1.0"

__all__ = ["run", "__version__"]


def __getattr__(name):
    # Imported lazily: pulling in run() imports torch, which is slow and
    # unnecessary for code that only wants __version__.
    if name == "run":
        from .chemembed_single_file import run
        return run
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
