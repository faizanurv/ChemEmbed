"""ChemEmbed — metabolite identification from MS/MS spectra via molecular embeddings.

A convolutional network predicts a 300-dimensional mol2vec embedding from a
spectrum, which is then matched by cosine similarity against a reference
database of pre-computed embeddings.

The trained models and the reference database are distributed separately from
this package because of their size; see the README for download instructions.
"""

from importlib.metadata import PackageNotFoundError, version as _version

try:
    # Read from the installed distribution rather than hardcoding, so the
    # version is declared once, in pyproject.toml. A literal here had already
    # gone stale: 1.1.1 shipped reporting __version__ == "1.1.0".
    __version__ = _version("chemembed")
except PackageNotFoundError:  # running from a source tree, not installed
    __version__ = "0.0.0.dev0"

__all__ = ["run", "__version__"]


def __getattr__(name):
    # Imported lazily: pulling in run() imports torch, which is slow and
    # unnecessary for code that only wants __version__.
    if name == "run":
        from .chemembed_single_file import run
        return run
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
