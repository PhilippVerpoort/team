from typing import Optional, Callable
import warnings
import re
from itertools import product

import pint
import numpy as np
import pandas as pd
from pandas.core.groupby import DataFrameGroupBy

from cet_units import Quantity, Unit

from .manipulations import CalcVariable, ExprAssignment, KeywordAssignment


# pandas dataframe accessor for groupby and perform methods
@pd.api.extensions.register_dataframe_accessor('team')
class TEAMAccessor:
    def __init__(self, df: pd.DataFrame):
        # Check that column axis has only one level.
        if df.columns.nlevels > 1:
            raise ValueError('Can only use .team accessor with team-like '
                             'dataframes that contain only one column layer.')

        # Warn if 'unfielded' column exists.
        if 'unfielded' in df.columns:
            warnings.warn("Having a column named 'unfielded' in the dataframe "
                          "may result in unexpected behaviour.")

        # Store dataframe as member field.
        self._df = df

    def fields(self,
               var_cols: str | list[str] | tuple[str] = 'variable',
               value_col: str = 'value',
               unit_col: str = 'unit') -> list[str]:
        """
        Return list of column names of fields.

        Parameters
        ----------
        var_cols : str or list[str] or tuple[str], optional
            Name of column containing the variables as strings. Default is
            `variable`. Multiple columns can be defined for advanced pivotting
            (with multi-level column index), e.g. when pivotting both
            `variable` and `period`.
        value_col : str, optional
            Name of the column containing the value. Default is `value`.
        unit_col : str, optional
            Name of the column containing the unit. Default is `unit`.

        Returns
        -------
            list of str
                A list of strings corresponding to the column names.
        """
        if isinstance(var_cols, str):
            var_cols = [var_cols]
        return [
            c for c in self._df
            if c not in ([value_col, unit_col] + var_cols)
        ]

    def explode(self,
                fields: Optional[str | list[str]] = None,
                var_cols: str | list[str] | tuple[str] = "variable",
                value_col: str = "value",
                unit_col: str = "unit") -> pd.DataFrame:
        """
        Explode rows with nan entries.

        Parameters
        ----------
        fields : str | list[str] | None
            The list of fields to explode.
        var_cols : str or list[str] or tuple[str], optional
            Name of column containing the variables as strings. Default is
            `variable`. Multiple columns can be defined for advanced pivotting
            (with multi-level column index), e.g. when pivotting both
            `variable` and `period`.
        value_col : str, optional
            Name of the column containing the value. Default is `value`.
        unit_col : str, optional
            Name of the column containing the unit. Default is `unit`.

        Returns
        -------
            pd.DataFrame
                The dataframe with nan entries in the respective fields
                exploded.
        """
        df = self._df
        fields = (
            self.fields(
                var_cols=var_cols,
                value_col=value_col,
                unit_col=unit_col,
            )
            if fields is None else
            [fields]
            if isinstance(fields, str) else
            fields
        )
        for field in fields:
            explodable = pd.Series(
                index=df.index,
                data=len(df)*[df[field].dropna().unique().tolist()],
            )
            df = (
                df
                .assign(**{field: df[field].fillna(explodable)})
                .explode(field)
            )

        return df.reset_index(drop=True)

    def groupby_fields(self,
                   var_cols: str | list[str] | tuple[str] = "variable",
                   value_col: str = "value",
                   unit_col: str = "unit",
                   **kwargs) -> DataFrameGroupBy:
        """
        Group by field columns (region, period, other...). Fields with
        rows that contain nan entries will be 'exploded' first.

        Parameters
        ----------
        var_cols : str or list[str] or tuple[str], optional
            Name of column containing the variables as strings. Default is
            `variable`. Multiple columns can be defined for advanced pivotting
            (with multi-level column index), e.g. when pivotting both
            `variable` and `period`.
        value_col : str, optional
            Name of the column containing the value. Default is `value`.
        unit_col : str, optional
            Name of the column containing the unit. Default is `unit`.
        kwargs
            Passed on to pd.DataFrame.groupby.

        Returns
        -------
            pd.DataFrameGroupBy
                The grouped dataframe rows.
        """
        if 'by' in kwargs:
            raise Exception("The 'by' argument is determined by team, you "
                            "cannot provide it manually.")
        fields = self.fields(
            var_cols=var_cols,
            value_col=value_col,
            unit_col=unit_col,
        )
        return self.explode().groupby(by=fields, **kwargs)

    def pivot_wide(self,
                   var_cols: str | list[str] | tuple[str] = "variable",
                   value_col: str = "value",
                   unit_col: str = "unit"):
        """
        Pivot dataframe wide, such that column names are variables.

        Parameters
        ----------
        var_cols : str or list[str] or tuple[str], optional
            Name of column containing the variables as strings. Default is
            `variable`. Multiple columns can be defined for advanced pivotting
            (with multi-level column index), e.g. when pivotting both
            `variable` and `period`.
        value_col : str, optional
            Name of the column containing the value. Default is `value`.
        unit_col : str, optional
            Name of the column containing the unit. Default is `unit`.

        Returns
        -------
            pd.DataFrame
                The original dataframe in pivot mode.
        """
        ret = self.explode()
        fields = self.fields(
            var_cols=var_cols,
            value_col=value_col,
            unit_col=unit_col,
        )

        # Create dummy field if non exists.
        if not fields:
            ret = ret.assign(unfielded=0)
            fields = ['unfielded']

        # Set variable columns.
        if isinstance(var_cols, str):
            var_cols = [var_cols]

        # Pivot dataframe.
        ret = ret.pivot(
            index=fields,
            columns=var_cols + [unit_col],
            values=value_col,
        )

        # Raise exception if duplicate cases exist.
        if ret.index.has_duplicates:
            raise ValueError('Performed pivot_wide on dataframe with '
                             'duplicate cases. Each variable should only be '
                             'defined once for each combination of field '
                             'values.')

        return ret.pint.quantify()

    def perform(self,
                func: Callable,
                pre_mapper: dict | Callable | None = None,
                post_mapper: dict | Callable | None = None,
                only_new: bool = True,
                dropna: bool = True,
                var_cols: str | list[str] | tuple[str] = 'variable',
                value_col: str = 'value',
                unit_col: str = 'unit',
                args : tuple = (),
                **kwargs):
        """
        Perform a manipulation by applying a function to the dataframe.

        Parameters
        ----------
        func : Callable
            Function to apply to dataframe. The function should accept a
            dataframe and return a dataframe. Additional arguments can be
            supplied through the key-word arguments.
        pre_mapper : dict-like or function, optional
            Dict-like or function transformations to apply to column names
            before passing dataframe as input to `func`.
        post_mapper : dict-like or function, optional
            Dict-like or function transformations to apply to column names
            before processing output from `func`.
        only_new : bool, optional
            Whether to only keep new variables. Default is True.
        dropna: bool, optional
            Whether to drop NaN entries resulting from the function call.
            Default is True.
        var_cols : str or list[str] or tuple[str], optional
            Name of column containing the variables as strings. Default is
            `variable`. Multiple columns can be defined for advanced pivotting
            (with multi-level column index), e.g. when pivotting both
            `variable` and `period`.
        value_col : str, optional
            Name of the column containing the value. Default is `value`.
        unit_col : str, optional
            Name of the column containing the unit. Default is `unit`.
        args : tuple
            Positional arguments to pass to `func` in addition to the data
            contained in the dataframe.
        **kwargs
            Additional keyword arguments to pass to `func` in addition to the
            data contained in the dataframe.

        Returns
        -------
            pd.DataFrame
                The dataframe that underwent the manipulation.

        """
        return self.perform_multi(
            [dict(
                func=func,
                pre_mapper=pre_mapper,
                post_mapper=post_mapper,
                args=args,
                **kwargs,
            )],
            only_new=only_new,
            dropna=dropna,
            var_cols=var_cols,
            value_col=value_col,
            unit_col=unit_col,
        )

    def perform_multi(self,
                funcs: list[dict],
                only_new: bool = True,
                dropna: bool = True,
                var_cols: str | list[str] | tuple[str] = "variable",
                value_col: str = "value",
                unit_col: str = "unit"):
        """
        Perform manipulations by applying multiple functions to the dataframe.

        Parameters
        ----------
        funcs : list[dict]
            A list of dictionaries, where each dictionary contains arguments
            `func` and (optionally) `pre_mapper`, `post_mapper`, `only_new`,
            `dropna`, `args`, and `kwargs` (same as `.team.perform()`).
        only_new : bool, optional
            Whether to only keep new variables. Default is True.
        dropna: bool, optional
            Whether to drop NaN entries resulting from the function call.
            Default is True.
        var_cols : str or list[str] or tuple[str], optional
            Name of column containing the variables as strings. Default is
            `variable`. Multiple columns can be defined for advanced pivotting
            (with multi-level column index), e.g. when pivotting both
            `variable` and `period`.
        value_col : str, optional
            Name of the column containing the value. Default is `value`.
        unit_col : str, optional
            Name of the column containing the unit. Default is `unit`.

        Returns
        -------
            pd.DataFrame
                The dataframe that underwent the manipulation(s).
        """
        # Convert var_cols to list if tuple.
        if isinstance(var_cols, tuple):
            var_cols = list(var_cols)
        
        # Pivot dataframe before manipulation.
        df_pivot = self.pivot_wide(
            var_cols=var_cols,
            value_col=value_col,
            unit_col=unit_col,
        )

        # Create list of column groups of variables and units.
        col_groups = (
            pd.Series(df_pivot.columns)
            .reset_index()
            .groupby(var_cols)
            .groups
        )

        # Raise exception in case of duplicate variables with different units.
        for col_name in col_groups:
            df_pivot_sub = df_pivot[col_name]
            if isinstance(df_pivot_sub, pd.Series):
                continue
            duplicate_indexes = (df_pivot_sub.notnull().sum(axis=1) > 1)
            if duplicate_indexes.any():
                warnings.warn(f"Duplicate entries in '{col_name}' for "
                              f"fields: {df_pivot.index[duplicate_indexes]}")

        # Loop over groups.
        return_list = []
        for col_ids in product(*col_groups.values()):
            df_pivot_group = df_pivot.iloc[:, list(col_ids)].dropna(how='all')
            original_index = df_pivot_group.index

            # Apply functions.
            results = []
            for func_specs in funcs:
                # Fetch from dict.
                func = func_specs.get('func')
                pre_mapper = func_specs.get('pre_mapper')
                post_mapper = func_specs.get('post_mapper')
                args = func_specs.get('args') or ()
                kwargs = {
                    k: v
                    for k, v in func_specs.items()
                    if k not in ['func', 'pre_mapper', 'post_mapper', 'args']
                }

                # Raise Exception if func not provided.
                if func is None:
                    raise Exception("No function provided to call.")
                elif not isinstance(func, Callable):
                    raise Exception("Function provided must be callable.")

                # Prepare data for function call.
                data = pd.concat([df_pivot_group] + results, axis=1)
                if pre_mapper is not None:
                    data = data.rename(columns=pre_mapper)

                # Call function and check results. Skip if result is empty.
                result = func(data, *args, **kwargs)
                if not isinstance(result, pd.DataFrame):
                    result = pd.DataFrame(result)
                if result.empty:
                    continue
                if not result.index.equals(original_index):
                    raise Exception('Manipulation may not change the index.')

                # Postprocess results and add to list.
                if post_mapper is not None:
                    result = result.rename(columns=post_mapper)
                results.append(result)

            # Combine results from functions, melt, and append to list.
            return_list.append(
                # Combine results
                pd.concat(results, axis=1)
                # Ensure that the axis label still exists before melt.
                .rename_axis(var_cols, axis=1)
                # Split units off into separate column.
                .pint.dequantify()
                # Melt.
                .melt(ignore_index=False)
                .reset_index()
            )

        # Combine groups into single dataframe.
        ret = pd.concat(return_list, ignore_index=True)

        # Convert pint objects to unit string and float value if any.
        if ret[value_col].dtype == 'object':
            locs = ret[value_col].apply(lambda x: isinstance(x, Quantity))
            ret.loc[locs, [unit_col, value_col]] = (
                ret.loc[locs, value_col].apply(
                    lambda x: pd.Series({unit_col: x.u,
                                         value_col: x.m})
                )
            )

        # Drop rows with nan entries in unit or value columns.
        if dropna:
            ret.dropna(subset=value_col, inplace=True)

        # Drop duplicates arising from multiple var-unit groups.
        ret.drop_duplicates(inplace=True)

        # Raise exception if index has duplicates after the above.
        fields = self.fields(
            var_cols=var_cols,
            value_col=value_col,
            unit_col=unit_col,
        )
        subset_cols = (
            fields + var_cols
            if isinstance(var_cols, list) else
            fields + [var_cols]
        )
        duplicates = ret.duplicated(subset=subset_cols)
        if duplicates.any():
            duplicate_labels = ret.loc[duplicates, subset_cols]
            raise Exception(f"Internal error: variables should only exist "
                            f"once per case: {duplicate_labels}")

        # Keep only new variables if requested.
        if not only_new:
            ret = pd.concat([self._df, ret], ignore_index=True)

        # Drop column called 'unfielded' if it exists.
        if 'unfielded' in ret.columns:
            ret = ret.drop(columns='unfielded')

        # Return dataframe.
        return ret.reset_index(drop=True)

    def calc_vars(self,
             *expr_assignments: ExprAssignment,
             only_new: bool = False,
             var_cols: str | list[str] | tuple[str] = "variable",
             value_col: str = "value",
             unit_col: str = "unit",
             **kw_assignments: KeywordAssignment):
        return self.perform(
            CalcVariable(*expr_assignments, **kw_assignments),
            only_new=only_new,
            var_cols=var_cols,
            value_col=value_col,
            unit_col=unit_col,
        )

    def varsplit(self,
                 cmd: Optional[str] = None,
                 regex: Optional[str] = None,
                 var_col_new: str = "variable",
                 var_new: str | bool = True,
                 keep_unmatched: bool = False) -> pd.DataFrame:
        """
        Split variable components separated by pipe characters into
        separate columns. The pattern must either be provided as
        `cmd` or as `regex`.

        Parameters
        ----------
        cmd : str
            A command to interpret into a regex.
        regex : str
            A direct regex.
        var_col_new : str
            (Optional) The name of the column where the new
            variable will be stored.
        var_new : str
            (Optional) The new variable name. True means, it is computed
            automatically. A string will be used to override it. False means
            that it is not added.
        keep_unmatched : bool
            Whether or not to keep unmatched rows.

        Returns
        -------
            pd.DataFrame
                The dataframe that contains the new split variables.
        """
        # Check that precisely one of the two arguments (either `cmd`
        # or `regex`) is provided.
        if cmd is not None and regex is not None:
            raise Exception('Only one of the two arguments may be provided: '
                            'cmd or regex.')
        if cmd is None and regex is None:
            raise Exception('Either a command or a regex string must be '
                            'provided.')

        # Check that target is in columns of dataframe.
        if var_col_new not in self._df.columns:
            raise Exception(f"Could not find column of name '{var_col_new}' in "
                            f"dataframe.")

        # Determine regex from cmd if necessary.
        if regex is None:
            regex = '^' + r'\|'.join([
                rf'(?P<{t[1:]}>[^|]*)' if t[0] == '?' else
                rf'(?P<{t[1:]}>.*)' if t[0] == '*' else
                re.escape(t)
                for t in cmd.split('|')
            ]) + '$'

        # Determine value of new variable column from arguments.
        if var_new is False:
            var_new = None
        elif var_new is True:
            if cmd is None:
                var_new = None
            else:
                var_new = '|'.join([
                    t for t in cmd.split('|')
                    if t[0] not in ('?', '*')
                ])

        # Create dataframe to be returned by applying regex to variable
        # column and dropping unmatched rows.
        matched = self._df[var_col_new].str.extract(regex)

        # Drop unmatched rows if requested.
        is_unmatched = matched.isna().any(axis=1)
        matched = matched.drop(index=matched.loc[is_unmatched].index)

        # Assign new variable column and drop if all are nan.
        if var_col_new not in matched:
            cond = matched.notnull().any(axis=1)
            matched[var_col_new] = self._df[var_col_new]
            matched.loc[cond, var_col_new] = var_new or np.nan
            if var_new is None:
                warnings.warn('New target column could not be set '
                              'automatically.')

        # Drop variable column if all nan.
        if matched[var_col_new].isnull().all():
            matched.drop(columns=var_col_new, inplace=True)

        # Combine with original dataframe.
        if keep_unmatched:
            df_combine = self._df.assign(**{
                var_col_new: lambda df: df[var_col_new].where(is_unmatched)
            })
        else:
            df_combine = self._df.loc[matched.index].drop(columns=var_col_new)
        ret = matched.combine_first(df_combine)

        # Sort columns.
        order = matched.columns.tolist() + self._df.columns.tolist()
        ret.sort_index(
            key=lambda cols: [order.index(c) for c in cols],
            axis=1,
            inplace=True,
        )

        # Return dataframe.
        return ret

    def varcombine(self,
                   cmd: str | Callable,
                   keep_cols: bool = False,
                   var_col: str = "variable") -> pd.DataFrame:
        """
        Combine columns into new variable (or other column).

        Parameters
        ----------
        cmd : str | Callable
            How the new variable (or other column) should be assembled.
        keep_cols : bool
            Whether to keep the used columns.
        var_col : str
            (Optional) The name of the target column. By default, this
            will be called `variable`.

        Returns
        -------
            pd.DataFrame
                The updated dataframe.
        """
        ret = self._df.assign(**{
            var_col: self._df.apply(lambda row:
                cmd.format(**row) if isinstance(cmd, str) else cmd(row),
                                    axis=1),
        })
        return ret if keep_cols else ret.filter([
            col for col in ret
            if col == var_col or
               (isinstance(cmd, Callable) or f"{{{col}}}" not in cmd)
        ])

    def sum_over(self,
                 over: str | list[str] | tuple[str],
                 value_col: str = "value",
                 unit_col: str = "unit"):
        """
        Sum up values over specified fields.

        Parameters
        ----------
        over : str | list[str] | tuple[str]
            The name of the field (column) to sum over.

        Returns
        -------
            pd.DataFrame
                The aggregated dataframe.
        """
        # Convert single string to list of strings.
        if isinstance(over, str):
            over = [over]

        def _helper_sum_units(group):
            q = (
                group
                .groupby(unit_col)
                .agg('sum')
                .reset_index()
                .apply(lambda row: row[value_col] * Unit(row[unit_col]), axis=1)
                .sum()
            )
            return(pd.Series({
                value_col: q.m,
                unit_col: f"{q.u:~}",
            }))

        group_cols = [
            c
            for c in self._df
            if c not in over and c not in [value_col, unit_col]
        ]

        if not group_cols:
            # implement
            pass

        return (
            self._df
            # group by all columns except unit, value, and 'over'
            .groupby(group_cols, dropna=False)
            # select only unit and value columns
            [[unit_col, value_col]]
            # apply summation
            .apply(
                lambda group:
                _helper_sum_units(group)
                if group[unit_col].nunique() > 1 else
                pd.Series({
                    value_col: group[value_col].sum(),
                    unit_col: group[unit_col].iloc[0],
                })
            )
            .reset_index()
        )

    def unit_to(self,
                other: str | Unit | dict[str, str | Unit],
                *contexts,
                var_cols: str | list[str] | tuple[str] = "variable",
                value_col: str = "value",
                unit_col: str = "unit",
                **ctx_kwargs):
        return self.__unit_convert(
            "to",
            other=other,
            contexts=contexts,
            var_cols=var_cols,
            value_col=value_col,
            unit_col=unit_col,
            ctx_kwargs=ctx_kwargs,
        )

    def unit_to_preferred(self,
                          other: str | Unit | dict[str, str | Unit],
                          *contexts,
                          var_cols: str | list[str] | tuple[str] = "variable",
                          value_col: str = "value",
                          unit_col: str = "unit",
                          **ctx_kwargs):
        return self.__unit_convert(
            "to_preferred",
            other=other,
            contexts=contexts,
            var_cols=var_cols,
            value_col=value_col,
            unit_col=unit_col,
            ctx_kwargs=ctx_kwargs,
        )

    def __unit_convert(self,
                       subfunc: str,
                       other: str | Unit | dict[str, str | Unit],
                       contexts: str | pint.Context,
                       var_cols: str | list[str] | tuple[str],
                       value_col: str,
                       unit_col: str,
                       ctx_kwargs: dict[str, str | pint.Context]):
        """
        Convert units in dataframe.

        Parameters
        ----------
        to : str | team_units.Unit | dict[str, str | team_units.Unit]
            The unit to convert to. This is either one unit for all rows
            or a dict that maps variables to units.

        Returns
        -------
            pd.DataFrame
                The dataframe with updated units.
        """
        # Check if to_unit is a single unit or a dict of units
        if not isinstance(other, dict):
            conv_factors = {}
            conv_units = {}
            if subfunc == "to_preferred":
                for from_unit in self._df[unit_col].unique():
                    other_q = getattr(Quantity(from_unit), subfunc)(other, *contexts, **ctx_kwargs)
                    conv_factors[from_unit] = other_q.m
                    conv_units[from_unit] = f"{other_q.u:~}"
                return self._df.assign(**{
                    value_col: self._df[value_col] * self._df[unit_col].map(conv_factors),
                    unit_col: self._df[unit_col].map(conv_units),
                })
            else:
                for from_unit in self._df[unit_col].unique():
                    other_q = getattr(Quantity(from_unit), subfunc)(other, *contexts, **ctx_kwargs)
                    conv_factors[from_unit] = other_q.m
                return self._df.assign(**{
                    value_col: self._df[value_col] * self._df[unit_col].map(conv_factors),
                    unit_col: str(other),
                })

        else:
            # Convert single string to list of strings.
            if isinstance(var_cols, str):
                var_cols = [var_cols]

            # Check that var_cols is valid.
            if not var_cols or any(c not in self._df for c in var_cols):
                raise Exception(f"Argument var_cols must contain valid column "
                                f"name(s). Found: {var_cols}. Must be one of: "
                                f"{", ".join(self._df.columns)}.")

            # Determine locations where conversion needs to take place.
            if len(var_cols) == 1:
                locs = self._df[var_cols[0]].isin(list(other))
            else:
                locs = self._df[var_cols].apply(tuple, axis=1).isin(list(other))

            # Define mappings for new units and conversion factors.
            mapping = (
                self._df.loc[locs, var_cols + [unit_col]]
                .drop_duplicates()
                .apply(
                    lambda row: pd.Series({
                        'index': tuple(row),
                        'conv_unit': other.get(tuple(row.iloc[0:-1]) if len(row) > 2 else row.iloc[0], row.iloc[-1]),
                        'conv_factor': getattr(Quantity(row.iloc[-1]), subfunc)(other.get(tuple(row.iloc[0:-1]) if len(row) > 2 else row.iloc[0], row.iloc[-1]), *contexts, **ctx_kwargs).m,
                    }),
                axis=1)
                .set_index('index')
            )
            conv_factors = mapping['conv_factor'].to_dict()
            conv_unit = mapping['conv_unit'].to_dict()

            # Apply mapping.
            mapping_key_col = self._df.loc[locs, var_cols + [unit_col]].apply(tuple, axis=1)
            new_values = self._df.loc[locs, value_col] * mapping_key_col.map(conv_factors)
            new_units = mapping_key_col.map(conv_unit)
            return self._df.assign(**{
                value_col: self._df[value_col].where(~locs, new_values),
                unit_col: self._df[unit_col].where(~locs, new_units),
            })
