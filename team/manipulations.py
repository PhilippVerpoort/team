import re
from typing import Optional, Callable, TypeAlias

import pandas as pd

from abc import abstractmethod


# define abstract manipulation class
class AbstractManipulation:
    @abstractmethod
    def perform(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    def _varsplit(self,
                  df: pd.DataFrame,
                  cmd: Optional[str] = None,
                  regex: Optional[str] = None) -> pd.DataFrame:
        # Check that precisely one of the two arguments (cmd and regex)
        # is provided.
        if cmd is not None and regex is not None:
            raise Exception('Only one of the two arguments may be provided: '
                            'cmd or regex.')
        if cmd is None and regex is None:
            raise Exception('Either a command or a regex string must be '
                            'provided.')

        # Determine regex from cmd if necessary.
        if regex is None:
            regex = '^' + r'\|'.join([
                rf'(?P<{t[1:]}>[^|]*)' if t[0] == '?' else
                rf'(?P<{t[1:]}>.*)' if t[0] == '*' else
                re.escape(t)
                for t in cmd.split('|')
            ]) + '$'

        # Extract new columns from existing.
        cols_extracted = df.columns.str.extract(regex)
        df_new = df[df.columns[cols_extracted.notnull().all(axis=1)]]
        df_new.columns = (
            pd.MultiIndex.from_frame(cols_extracted.dropna())
            if len(cols_extracted.columns) > 1 else
            cols_extracted.dropna().iloc[:, 0]
        )

        # Return new dataframe.
        return df_new


# new variable can be calculated through expression assignments or
# keyword assignments expression assignments are of form
# "`a` = `b` + `c`" and are based on the pandas eval functionality
# keyword assignment must be int, float, string, or a function to
# be called to assign to the variable defined as key
ExprAssignment: TypeAlias = str
KeywordAssignment: TypeAlias = int | float | str | Callable


# Generic manipulation for calculating variables.
class CalcVariable(AbstractManipulation):
    _expr_assignments: tuple[ExprAssignment]
    _kw_assignments: dict[str, KeywordAssignment]

    def __init__(self,
                 *expr_assignments: ExprAssignment,
                 **kw_assignments: KeywordAssignment):
        self._expr_assignments = expr_assignments
        self._kw_assignments = kw_assignments

        # check all supplied arguments are valid
        for expr_assignment in self._expr_assignments:
            if not isinstance(expr_assignment, str):
                raise Exception(f"Expression assignments must be of type str, "
                                f"but found: {type(expr_assignment)}")
        for kw_assignment in self._kw_assignments.values():
            if not (isinstance(kw_assignment, int | float | str) or
                    callable(kw_assignment)):
                raise Exception(f"Keyword assignments must be of type int, "
                                f"float, string, or callable, but found: "
                                f"{type(kw_assignment)}")

    def perform(self, df: pd.DataFrame) -> pd.DataFrame:
        for expr_assignment in self._expr_assignments:
            df.eval(expr_assignment, inplace=True, engine='python')
        df = df.assign(**self._kw_assignments)
        return df


# Generic manipulation for applying functions.
class Apply(AbstractManipulation):
    _callable: Callable | None
    _callables: dict[str, Callable]
    def __init__(self,
                 callable: Callable | None = None,
                 **kwargs: Callable):
        if (callable and kwargs) or (not callable and not kwargs):
            ex_msg = ('Please provide either one callable for all columns or '
                      'specifiy callables for different columns by their names '
                      'as keyword arguments.')
            if callable and kwargs:
                ex_msg += ' You cannot provide both at the same time.'
            raise Exception(ex_msg)

        self._callable = callable
        self._callables = kwargs

    def perform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._callable:
            return df.apply(self._callable)
        else:
            for col_id, col_callable in self._callables.items():
                df = df.assign(col_id=col_callable(df[col_id]))
            return df
