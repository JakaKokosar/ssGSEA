# Single Sample GSEA analysis

Single-sample geneset enrichment analysis (ssGSEA) is a single-sample extension of the GSEA algorithm. It calculates a separate enrichment score for each sample and gene set pairing [[1]](#1).

## Install the latest version

    pip install single_sample_gsea

## Usage

```python
>>> from single_sample_gsea import ss_gsea

>>> gene_sets = {
    "gs1": {"gene2", "gene3"},
    "gs2": {"gene1", "gene4"},
    }

>>> data = {
    "gene": ["gene1", "gene2", "gene3", "gene4", "gene5"],
    "sample-1": [1, 3, 4, 7, 32],
    "sample-2": [25, 4, 6, 18, 1],
    }
>>> data = pd.DataFrame(data).set_index("gene")

>>> ss_gsea(data, gene_sets)
               gs1       gs2
sample-1 -1.333333 -0.962974
sample-2 -1.333333  2.543214
```

## References

<a id="1">[1]</a>
Barbie, D., Tamayo, P., Boehm, J. et al. Systematic RNA interference reveals that oncogenic KRAS-driven cancers require TBK1. Nature 462, 108–112 (2009). <https://doi.org/10.1038/nature08460>
