import pandas as pd
import numpy as np
import numba as nb
import scipy.stats as ss


@nb.njit(parallel=True)
def _isin(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_as1d = a.ravel()
    n = len(a_as1d)
    b_ = set(b)

    result = np.full(n, False)
    for i in nb.prange(n):
        result[i] = a_as1d[i] in b_

    return result.reshape(a.shape)


def ss_gsea(
    df: pd.DataFrame, gene_sets: dict, alpha: float = 0.25, callback=None
) -> pd.DataFrame:
    x = df.values

    # rank normalize
    x = ss.rankdata(x, axis=0)

    # Get indexes of ranks in decreasing order
    ranks_decreasing_indexes = np.argsort(x, axis=0)[::-1]

    # z-transform
    x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
    x = np.abs(x)

    # rank weight
    x = x**alpha

    # Sort ranks in decreasing order
    ranks_sorted_decreasing = np.take_along_axis(x, ranks_decreasing_indexes, axis=0)

    scores = {}
    for gs_name, gs_genes in gene_sets.items():
        # indexes of signature genes in the data
        gs_genes_index = np.where(df.index.isin(gs_genes))[0]

        # Get positions of signatures genes in sorted ranks
        gs_genes_to_rank_indexes = _isin(ranks_decreasing_indexes, gs_genes_index)

        # Compute ECDF for signature genes
        signature_ecdf = ranks_sorted_decreasing * gs_genes_to_rank_indexes
        signature_ecdf = signature_ecdf / np.sum(signature_ecdf, axis=0)
        signature_ecdf = np.cumsum(signature_ecdf, axis=0)

        # Compute ECDF for the remaining gene
        non_signature_ecdf = ~gs_genes_to_rank_indexes / np.sum(
            ~gs_genes_to_rank_indexes, axis=0
        )
        non_signature_ecdf = np.cumsum(non_signature_ecdf, axis=0)

        # sum of the difference between ECDFs
        enrichment_score = np.sum(signature_ecdf - non_signature_ecdf, axis=0)

        # save result for corresponding gene set
        scores[gs_name] = enrichment_score

        if callback is not None:
            callback()

    df_results = pd.DataFrame(scores)
    df_results.index = df.columns
    return df_results


if __name__ == "__main__":
    gene_sets = {
        "gs1": {"gene2", "gene3"},
        "gs2": {"gene1", "gene4"},
    }

    data = {
        "gene": ["gene1", "gene2", "gene3", "gene4", "gene5"],
        "sample-1": [1, 3, 4, 7, 32],
        "sample-2": [25, 4, 6, 18, 1],
    }

    data = pd.DataFrame(data).set_index("gene")
    result = ss_gsea(data, gene_sets)
    print(result)
