.. _Installation:

Installation and Setup
================================================================================

Installing SIGnature
--------------------------------------------------------------------------------

The SIGnature API is under activate development. The latest development API
can be downloaded from `GitHub <https://github.com/Genentech/SIGnature.git>`__
and installed as follows:

::

    git clone https://github.com/genentech/signature.git
    cd signature
    pip install -e .

Downloading precalculated attributions
--------------------------------------------------------------------------------

Download the cell type-level attributions from Zenodo:
https://zenodo.org/communities/signature/

Downloading models
--------------------------------------------------------------------------------

Download the SCimilarity model from Zenodo:
https://zenodo.org/records/15729925

Conda environment setup
--------------------------------------------------------------------------------

To install SIGnature in a [Conda](https://docs.conda.io) environment
we recommend this environment setup:

:download:`Download environment file <_static/environment.yaml>`

.. literalinclude:: _static/environment.yaml
  :language: YAML

Followed by installing the ``signature`` package, as above.
