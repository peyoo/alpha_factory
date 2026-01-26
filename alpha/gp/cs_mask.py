import polars as pl
from polars import Expr
from polars_ta.wq import cs_rank



def cs_rank_mask(x: Expr,mask_col:str = 'MASK',  pct: bool = True) -> Expr:
    condition = pl.col(mask_col)
    return cs_rank(pl.when(condition).then(x).otherwise(None), pct)


CUSTOM_OPERATORS = {
    'cs_rank_mask': cs_rank_mask
}