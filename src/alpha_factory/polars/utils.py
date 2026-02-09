from alpha_factory.polars.cs import cs_mad_zscore_mask, cs_demean_mask, cs_rank_mask, cs_qcut_mask

CUSTOM_OPERATORS = {
    "cs_mad_zscore_mask": cs_mad_zscore_mask,
    'cs_rank_mask': cs_rank_mask,
    'cs_demean_mask': cs_demean_mask,
    'cs_qcut_mask': cs_qcut_mask,
}
