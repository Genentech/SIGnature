SIGnature: Scoring the Importance of Genes
================================================================================

**SIGnature** is a Python package that empowers researchers to rapidly query gene sets across diverse single-cell RNA sequencing (scRNA-seq) datasets through precomputed gene attribution scores.

Beyond querying capabilities, SIGnature also enables the generation of attribution scores on novel scRNA-seq data, allowing seamless integration with our collection of annotated studies.

Documentation
--------------------------------------------------------------------------------

Tutorials and API documentation can be found at:
https://genentech.github.io/SIGnature/index.html


Download & Install
--------------------------------------------------------------------------------

The SIGnature API is under activate development. The latest development API can be downloaded and installed as follows::

    git clone https://github.com/genentech/signature.git
    cd signature
    pip install -e .

Precomputed attribution scores can be downloaded from Zenodo:
https://zenodo.org/communities/signature/

To generate attribution scores on new data, helper files can be downloaded from Zenodo:
https://zenodo.org/records/17903196

SIGnature currently supports calculating attributions using the following models:


1. `SCimilarity <https://doi.org/10.1038/s41586-024-08411-y>`_: pretrained weights included in helper files, but can also be downloaded here: https://zenodo.org/records/15729925
2. `scFoundation <https://doi.org/10.1038/s41592-024-02305-7>`_: pretrained weights can be downloaded here: https://huggingface.co/genbio-ai/scFoundation/tree/main
3. `scVI <https://doi.org/10.1038/s41592-018-0229-2>`_: pretrained weights for scVI models trained on CZI Census data can be downloaded here: https://cellxgene.cziscience.com/census-models
4. `SSL-scTab <https://doi.org/10.1038/s42256-024-00934-3>`_: pretrained weights for self-supervised learning models trained on scTab data can be downloaded here: https://huggingface.co/TillR/sc_pretrained/tree/main


Citation
--------------------------------------------------------------------------------

To cite SIGnature in publications please use:

**Foundation Model Attributions Reveal Shared Inflammatory Program Across Diseases**

Maxwell P. Gold, Miguel Reyes, Nathaniel Diamant, Tony Kuo, Ehsan Hajiramezanali, Jane W. Newburger, Mary Beth F. Son, Pui Y. Lee, Gabriele Scalia, Aicha BenTaieb, Sharookh B. Kapadia, Anupriya Tripathi, Héctor Corrada Bravo, Graham Heimberg\*, Tommaso Biancalani\* bioRxiv 2025.06.14.659567; doi: https://doi.org/10.1101/2025.06.14.659567
