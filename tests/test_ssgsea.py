import pandas as pd


from single_sample_gsea import ss_gsea


def test_ss_gsea():
    gene_sets = {
        "gs1": {"gene2", "gene3"},
        "gs2": {"gene1", "gene4"},
        # 'gs3': {'gene1', 'gene4'},
    }

    data = {
        "gene": ["gene1", "gene2", "gene3", "gene4", "gene5"],
        "sample-1": [1, 3, 4, 7, 32],
        "sample-2": [25, 4, 6, 18, 1],
        # 'sample-3': [25, 4, 6, 18, 4],
    }

    data = pd.DataFrame(data).set_index("gene")

    result = ss_gsea(data, gene_sets)

    assert isinstance(result, pd.DataFrame)
    assert result.index.to_list() == ["sample-1", "sample-2"]

    assert len(result.columns) == 2
    assert result.columns.to_list() == ["gs1", "gs2"]
