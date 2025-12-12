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

To generate attribution scores on new data, helper files can be downloaded from Zenodo:
https://zenodo.org/records/17903196

SIGnature currently supports calculating attributions using the following models:


1. `SCimilarity <https://doi.org/10.1038/s41586-024-08411-y>`_: pretrained weights included in helper files, but can also be downloaded here: https://zenodo.org/records/15729925
2. `scFoundation <https://doi.org/10.1038/s41592-024-02305-7>`_: pretrained weights can be downloaded here: https://huggingface.co/genbio-ai/scFoundation/tree/main
3. `scVI <https://doi.org/10.1038/s41592-018-0229-2>`_: pretrained weights for scVI models trained on CZI Census data can be downloaded here: https://cellxgene.cziscience.com/census-models
4. `SSL-scTab <https://doi.org/10.1038/s42256-024-00934-3>`_: pretrained weights for self-supervised learning models trained on scTab data can be downloaded here: https://huggingface.co/TillR/sc_pretrained/tree/main


Conda environment setup
--------------------------------------------------------------------------------

To install SIGnature in a [Conda](https://docs.conda.io) environment
we recommend this environment setup:

:download:`Download environment file <_static/environment.yaml>`

.. literalinclude:: _static/environment.yaml
  :language: YAML

Followed by installing the ``signature`` package, as above.
