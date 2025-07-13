from typing import Callable, Optional, TypeAlias
import warnings
from re import escape, findall
from itertools import product

import pint
import numpy as np
import pandas as pd
from pandas.core.groupby import DataFrameGroupBy

from cet_units import Quantity, Unit


def _sort_cols(
    df: pd.DataFrame,
    var_cols: str | list[str],
    unit_col: str,
    value_col: str,
    fields_order: list[str] = None,
):
    # Preprocess arguments.
    var_cols = var_cols if isinstance(var_cols, list) else [var_cols]
    fields_order = fields_order or []

    # Determine base columns and known columns.
    base_cols = var_cols + [unit_col, value_col]
    known_cols = fields_order + base_cols

    # Determine other columns.
    other_cols = [c for c in df.columns if c not in known_cols]

    # Sort other columns.
    other_cols.sort()

    return df[fields_order + other_cols + [c for c in base_cols if c in df]]

def _sort_rows(df: pd.DataFrame):
    return df


ExprAssignment: TypeAlias = str
KeywordAssignment: TypeAlias = int | float | str | Callable


def _calculate(
    df: pd.DataFrame,
    *expr_assignments: ExprAssignment,
    **kw_assignments: KeywordAssignment,
):
    expr_cols: list[str] = []
    for expr_assignment in expr_assignments:
        # This regex matches column assignments like 'col = ...' or
        # '`col name` = ...'.
        matches = findall(r'`([^`]+)`\s*=\s*|(\w+)\s*=', expr_assignment)

        # Flatten the matches and filter out None values.
        expr_cols += [match[0] if match[0] else match[1] for match in matches]

        # Calculate new columns.
        df.eval(expr_assignment, inplace=True, engine="python")
    for val in kw_assignments.values():
        if isinstance(val, str) and "=" in val:
            raise Exception("Keyword assignment with string may not contain "
                            "additional assignments.")
    kw_assignments = {
        k: (
            lambda df: df.eval(v)
            if isinstance(v, str)
            else v(df)
            if isinstance(v, Callable)
            else v
        )
        for k, v in kw_assignments.items()
    }
    df = df.assign(**kw_assignments)
    return df[expr_cols + list(kw_assignments)]


@pd.api.extensions.register_dataframe_accessor("team")
class TEAMAccessor:
    def __init__(self, df: pd.DataFrame):
        # Check that column axis has only one level.
        if df.columns.nlevels > 1:
            raise ValueError(
                "Can only use .team accessor with team-like dataframes that "
                "contain only one column layer."
            )

        # Warn if 'unfielded' column exists.
        if "unfielded" in df.columns:
            warnings.warn(
                "Having a column named 'unfielded' in the dataframe "
                "may result in unexpected behaviour."
            )

        # Store dataframe as member field.
        self._df = df

    def fields(
        self,
        var_cols: str | list[str] | tuple[str] = "variable",
        value_col: str = "value",
        unit_col: str = "unit",
    ) -> list[str]:
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
            c
            for c in self._df
            if c not in ([value_col, unit_col] + var_cols)
        ]

    def sort_columns(
        self,
        fields_order: list[str] | None = None,
        var_cols: str | list[str] | tuple[str] = "variable",
        value_col: str = "value",
        unit_col: str = "unit",
    ) -> pd.DataFrame:
        return _sort_cols(
            df=self._df,
            fields_order=fields_order,
            var_cols=var_cols,
            value_col=value_col,
            unit_col=unit_col,
        )

    def sort_rows(
        self,
        by: str | list[str] | None = None,
        var_cols: str | list[str] | tuple[str] = "variable",
        value_col: str = "value",
        unit_col: str = "unit",
    ):
        if by is None:
            var_cols = var_cols if isinstance(var_cols, list) else [var_cols]
            base_cols = var_cols + [value_col, unit_col]

            by = [c for c in self._df if c not in base_cols]
            if all(v in self._df for v in var_cols):
                by += var_cols
        elif not by:
            raise Exception("Argument `by` may not be an empty list or str.")

        return self._df.sort_values(by=by).reset_index(drop=True)

    def explode(
        self,
        fields: Optional[str | list[str]] = None,
        var_cols: str | list[str] | tuple[str] = "variable",
        value_col: str = "value",
        unit_col: str = "unit",
    ) -> pd.DataFrame:
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

        if fields is None:
            fields = self.fields(
                var_cols=var_cols,
                value_col=value_col,
                unit_col=unit_col,
            )
        elif isinstance(fields, str):
            fields = [fields]

        for field in fields:
            explodable = pd.Series(
                index=df.index,
                data=len(df) * [df[field].dropna().unique().tolist()],
            )
            df = (
                df
                .assign(**{field: df[field].fillna(explodable)})
                .explode(field)
            )

        return df.reset_index(drop=True)

    def groupby_fields(
        self,
        var_cols: str | list[str] | tuple[str] = "variable",
        value_col: str = "value",
        unit_col: str = "unit",
        **kwargs,
    ) -> DataFrameGroupBy:
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
        if "by" in kwargs:
            raise Exception(
                "The 'by' argument is determined by team, you "
                "cannot provide it manually."
            )
        fields = self.fields(
            var_cols=var_cols,
            value_col=value_col,
            unit_col=unit_col,
        )
        return (
            self
            .explode()
            .groupby(by=fields, **kwargs)
        )

    def pivot_wide(
        self,
        var_cols: str | list[str] | tuple[str] = "variable",
        value_col: str = "value",
        unit_col: str = "unit",
    ):
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
        ret = self.explode(
            var_cols=var_cols,
            value_col=value_col,
            unit_col=unit_col,
        )
        fields = self.fields(
            var_cols=var_cols,
            value_col=value_col,
            unit_col=unit_col,
        )

        # Create dummy field if non exists.
        if not fields:
            ret = ret.assign(unfielded=0)
            fields = ["unfielded"]

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
            raise ValueError(
                "Performed pivot_wide on dataframe with "
                "duplicate cases. Each variable should only be "
                "defined once for each combination of field "
                "values."
            )

        return ret.pint.quantify()

    def perform(
        self,
        func: Callable,
        pre_mapper: dict | Callable | None = None,
        post_mapper: dict | Callable | None = None,
        using: pd.DataFrame | list[pd.DataFrame] | None = None,
        only_new: bool = False,
        fill_missing: bool = False,
        dropna: bool = True,
        var_cols: str | list[str] | tuple[str] = "variable",
        value_col: str = "value",
        unit_col: str = "unit",
        args: tuple = (),
        **kwargs,
    ):
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
            Whether to only keep new variables. Default is False.
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
            [
                dict(
                    func=func,
                    pre_mapper=pre_mapper,
                    post_mapper=post_mapper,
                    args=args,
                    **kwargs,
                )
            ],
            using=using,
            only_new=only_new,
            fill_missing=fill_missing,
            dropna=dropna,
            var_cols=var_cols,
            value_col=value_col,
            unit_col=unit_col,
        )

    def perform_multi(
        self,
        funcs: list[dict],
        using: pd.DataFrame | list[pd.DataFrame] | None = None,
        only_new: bool = False,
        dropna: bool = True,
        fill_missing: bool = False,
        var_cols: str | list[str] | tuple[str] = "variable",
        value_col: str = "value",
        unit_col: str = "unit",
    ):
        """
        Perform manipulations by applying multiple functions to the dataframe.

        Parameters
        ----------
        funcs : list[dict]
            A list of dictionaries, where each dictionary contains arguments
            `func` and (optionally) `pre_mapper`, `post_mapper`, `only_new`,
            `dropna`, `args`, and `kwargs` (same as `.team.perform()`).
        only_new : bool, optional
            Whether to only keep new variables. Default is False.
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
        elif not isinstance(var_cols, list):
            var_cols = [var_cols]

        # Combine dataframe `using` with the one associated with this object.
        if using is not None:
            if isinstance(using, pd.DataFrame):
                using = pd.concat([using])
            elif not (
                isinstance(using, list)
                and all(isinstance(u, pd.DataFrame) for u in using)
            ):
                raise Exception(
                    "Argument `using` must be dataframe or list of dataframes."
                )

            df = pd.concat([self._df, using])
        else:
            df = self._df

        # Create dummy field if non exists.
        if not [c for c in df if c not in (var_cols + [unit_col, value_col])]:
            df = df.assign(unfielded=0)

        # Explode.
        df = df.team.explode(
            var_cols=var_cols,
            unit_col=unit_col,
            value_col=value_col,
        )

        # Group by all fields, by variable columns, and by unit. Verify
        # integrity to ensure that there are now duplicates.
        df_grouped = df.set_index([
            c
            for c in df.columns
            if c != value_col
        ], verify_integrity=True)[value_col]

        # Pivot dataframe and set pint units.
        df_pivot = (
            df_grouped
            .unstack(var_cols + [unit_col])
        )

        # Get a presence matrix.
        presence = (
            pd.Series(True, index=df_grouped.index)
            .unstack(var_cols + [unit_col], fill_value=False)
        )

        # Determine presence groups.
        pivot_groups = (
            presence
            .groupby([c for c in presence.columns])
            .groups
        )

        # Loop over pivot groups.
        return_list = []
        for pivot_group_cols, pivot_group_row in pivot_groups.items():
            # Get column names from boolean indexer.
            pivot_group_cols = (
                df_pivot
                .columns[pd.Series(pivot_group_cols, index=df_pivot.columns)]
            )

            # Make sure that there are no columns with more than one unit
            # defined per variable.
            if not pivot_group_cols.droplevel(level=unit_col).is_unique:
                raise Exception(
                    "Variable defined with multiple units: "
                    + df_pivot.loc[pivot_group_row]
                )

            # Get the pivot group by selecting relevant rows and columns and
            # using pint to set the dtype of the column from the unit level.
            df_pivot_group = (
                df_pivot
                .loc[pivot_group_row, pivot_group_cols]
                .pint.quantify()
            )

            # Keep a copy of the index to ensure that it is not changed.
            original_index = df_pivot_group.index

            # If requested, keep a copy of the original pivot group.
            if not only_new:
                df_pivot_group_original = df_pivot_group.copy()

            # Insert missing columns. This will be necessary in case the
            # performed actions operate on column not present in this group.
            if fill_missing:
                cols = (
                    df_pivot
                    .columns
                    .to_frame()
                    .droplevel(unit_col)[unit_col]
                    .to_dict()
                )
                missing_cols = {
                    col_id: col_unit
                    for col_id, col_unit in cols.items()
                    if col_id not in df_pivot_group.columns
                }
                for col_id, col_unit in missing_cols.items():
                    df_pivot_group[col_id] = pd.Series(
                        np.nan,
                        dtype=f"pint[{col_unit}][float64]"
                    )

            # Apply functions.
            for func_specs in funcs:
                # Fetch from dict.
                func = func_specs.get("func")
                pre_mapper = func_specs.get("pre_mapper")
                post_mapper = func_specs.get("post_mapper")
                args = func_specs.get("args") or ()
                kwargs = {
                    k: v
                    for k, v in func_specs.items()
                    if k not in ["func", "pre_mapper", "post_mapper", "args"]
                }

                # Raise Exception if func not provided.
                if func is None:
                    raise Exception("No function provided to call.")
                elif not isinstance(func, Callable):
                    raise Exception("Function provided must be callable.")

                # Prepare data for function call.
                if pre_mapper is not None:
                    df_pivot_group = df_pivot_group.rename(columns=pre_mapper)

                # Call function and check results. Skip if result is empty.
                df_pivot_group = func(df_pivot_group, *args, **kwargs)
                if isinstance(df_pivot_group, dict):
                    df_pivot_group = pd.DataFrame(df_pivot_group)
                elif not isinstance(df_pivot_group, pd.DataFrame):
                    raise Exception("Return type must be a dict or dataframe.")
                if not df_pivot_group.index.equals(original_index):
                    raise Exception("Manipulation may not change the index.")
                if df_pivot_group.empty:
                    raise Exception("Returned dataframe may not be empty.")

                # Postprocess results and add to list.
                if post_mapper is not None:
                    df_pivot_group = df_pivot_group.rename(columns=post_mapper)

            # Keep only new variables if requested.
            if not only_new:
                cols_not_overridden = [
                    col
                    for col in df_pivot_group_original.columns
                    if col not in df_pivot_group.columns
                ]
                if cols_not_overridden:
                    df_pivot_group = pd.concat([
                        df_pivot_group_original[cols_not_overridden],
                        df_pivot_group,
                    ], axis=1)

            # Drop dummy columns.
            if fill_missing and missing_cols:
                df_pivot_group.drop(
                    columns=list(missing_cols),
                    errors="ignore",
                    inplace=True,
                )

            # Combine results from functions, melt, and append to list.
            if not (df_pivot_group.empty or df_pivot.isnull().all().all()):
                return_list.append(
                    df_pivot_group
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
        if ret[value_col].dtype == "object":
            locs = ret[value_col].apply(lambda x: isinstance(x, Quantity))
            ret.loc[locs, [unit_col, value_col]] = ret.loc[
                locs, value_col
            ].apply(lambda x: pd.Series({unit_col: x.u, value_col: x.m}))

        # Drop rows with nan entries in unit or value columns.
        if dropna:
            ret.dropna(subset=value_col, inplace=True)

        # Drop duplicates arising from multiple var-unit groups.
        ret.drop_duplicates(inplace=True)

        # Raise exception if index has duplicates after the above.
        fields = df.team.fields(
            var_cols=var_cols,
            value_col=value_col,
            unit_col=unit_col,
        )
        duplicates = ret.duplicated(subset=fields + var_cols)
        if duplicates.any():
            duplicate_labels = ret.loc[duplicates, subset_cols]
            warnings.warn(
                f"Internal error: variables should only exist "
                f"once per case: {duplicate_labels}"
            )

        # Drop column called 'unfielded' if it exists.
        if "unfielded" in ret.columns:
            ret = ret.drop(columns="unfielded")

        # Return.
        return ret.reset_index(drop=True)

    def calculate(
        self,
        *expr_assignments: ExprAssignment,
        only_new: bool = False,
        fill_missing: bool = False,
        var_cols: str | list[str] | tuple[str] = "variable",
        value_col: str = "value",
        unit_col: str = "unit",
        **kw_assignments: KeywordAssignment,
    ):
        """
        This accessor method allows calculating new variables without defining
        a complete function to be used with `.perform()`. New variables can be
        calculated through expression assignments or keyword assignments.
        Expression assignments are of form "`a` = `b` + `c`" and are based on
        the pandas eval functionality. Keyword assignment must be int,float,
        string, or a function to be called to assign to the variable defined
        as key.

        Parameters
        ----------
        expr_assignments : str
        only_new : bool
            Whether only new values should be kept. Default is False.
        var_cols : str | list[str] | tuple[str]
            The name(s) of the varaible column(s) to use. Default is
            `variable`.
        value_col : str
            The name of the value column to use. Default is `value`.
        unit_col : str
            The name of the unit column to use. Default is `unit`.
        kw_assignments : int | float | str | Callable
            A value to assign to a column via keyword assignments.

        Returns
        -------
        pd.DataFrame
            The new dataframe with the newly calculated variables.

        """
        return self.perform(
            lambda df: _calculate(df, *expr_assignments, **kw_assignments),
            only_new=only_new,
            fill_missing=fill_missing,
            var_cols=var_cols,
            value_col=value_col,
            unit_col=unit_col,
        )

    def varsplit(
        self,
        cmd: Optional[str] = None,
        regex: Optional[str] = None,
        var_col_new: str = "variable",
        var_new: str | bool = True,
        keep_unmatched: bool = False,
    ) -> pd.DataFrame:
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
            raise Exception(
                "Only one of the two arguments may be provided: cmd or regex."
            )
        if cmd is None and regex is None:
            raise Exception(
                "Either a command or a regex string must be provided."
            )

        # Check that target is in columns of dataframe.
        if var_col_new not in self._df.columns:
            raise Exception(
                f"Could not find column of name '{var_col_new}' in dataframe."
            )

        # Determine regex from cmd if necessary.
        if regex is None:
            regex = (
                "^"
                + r"\|".join(
                    [
                        rf"(?P<{t[1:]}>[^|]*)"
                        if t[0] == "?"
                        else rf"(?P<{t[1:]}>.*)"
                        if t[0] == "*"
                        else escape(t)
                        for t in cmd.split("|")
                    ]
                )
                + "$"
            )

        # Determine value of new variable column from arguments.
        if var_new is False:
            var_new = None
        elif var_new is True:
            if cmd is None:
                var_new = None
            else:
                var_new = "|".join(
                    [t for t in cmd.split("|") if t[0] not in ("?", "*")]
                )

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
                warnings.warn(
                    "New target column could not be set automatically."
                )

        # Drop variable column if all nan.
        if matched[var_col_new].isnull().all():
            matched.drop(columns=var_col_new, inplace=True)

        # Combine with original dataframe.
        if keep_unmatched:
            df_combine = self._df.assign(
                **{var_col_new: lambda df: df[var_col_new].where(is_unmatched)}
            )
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

    def varcombine(
        self,
        cmd: str | Callable,
        keep_cols: bool = False,
        var_col: str = "variable",
    ) -> pd.DataFrame:
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
        ret = self._df.assign(
            **{
                var_col: self._df.apply(
                    lambda row: cmd.format(**row)
                    if isinstance(cmd, str)
                    else cmd(row),
                    axis=1,
                ),
            }
        )
        return (
            ret
            if keep_cols
            else ret.filter(
                [
                    col
                    for col in ret
                    if col == var_col
                    or (isinstance(cmd, Callable) or f"{{{col}}}" not in cmd)
                ]
            )
        )

    def sum_over(
        self,
        over: str | list[str] | tuple[str],
        value_col: str = "value",
        unit_col: str = "unit",
    ):
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
                group.groupby(unit_col)
                .agg("sum")
                .reset_index()
                .apply(
                    lambda row: row[value_col] * Unit(row[unit_col]), axis=1
                )
                .sum()
            )
            return pd.Series(
                {
                    value_col: q.m,
                    unit_col: f"{q.u:~}",
                }
            )

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
            .groupby(group_cols, dropna=False)[
                # select only unit and value columns
                [unit_col, value_col]
            ]
            # apply summation
            .apply(
                lambda group: _helper_sum_units(group)
                if group[unit_col].nunique() > 1
                else pd.Series(
                    {
                        value_col: group[value_col].sum(),
                        unit_col: group[unit_col].iloc[0],
                    }
                )
            )
            .reset_index()
        )

    def unit_to(
        self,
        other: str | Unit | dict[str, str | Unit],
        *contexts,
        var_cols: str | list[str] | tuple[str] = "variable",
        value_col: str = "value",
        unit_col: str = "unit",
        **ctx_kwargs,
    ):
        return self.__unit_convert(
            "to",
            other=other,
            contexts=contexts,
            var_cols=var_cols,
            value_col=value_col,
            unit_col=unit_col,
            ctx_kwargs=ctx_kwargs,
        )

    def unit_to_preferred(
        self,
        other: str | Unit | dict[str, str | Unit],
        *contexts,
        var_cols: str | list[str] | tuple[str] = "variable",
        value_col: str = "value",
        unit_col: str = "unit",
        **ctx_kwargs,
    ):
        return self.__unit_convert(
            "to_preferred",
            other=other,
            contexts=contexts,
            var_cols=var_cols,
            value_col=value_col,
            unit_col=unit_col,
            ctx_kwargs=ctx_kwargs,
        )

    def __unit_convert(
        self,
        subfunc: str,
        other: str | Unit | dict[str, str | Unit],
        contexts: str | pint.Context,
        var_cols: str | list[str] | tuple[str],
        value_col: str,
        unit_col: str,
        ctx_kwargs: dict[str, str | pint.Context],
    ):
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
                    from_unit_safe = from_unit or "dimensionless"
                    other_q = getattr(Quantity(from_unit_safe), subfunc)(
                        other, *contexts, **ctx_kwargs
                    )
                    conv_factors[from_unit] = other_q.m
                    conv_units[from_unit] = f"{other_q.u:~}"
                return self._df.assign(
                    **{
                        value_col: self._df[value_col]
                        * self._df[unit_col].map(conv_factors),
                        unit_col: self._df[unit_col].map(conv_units),
                    }
                )
            else:
                for from_unit in self._df[unit_col].unique():
                    from_unit_safe = from_unit or "dimensionless"
                    other_q = getattr(Quantity(from_unit_safe), subfunc)(
                        other, *contexts, **ctx_kwargs
                    )
                    conv_factors[from_unit] = other_q.m
                return self._df.assign(
                    **{
                        value_col: self._df[value_col]
                        * self._df[unit_col].map(conv_factors),
                        unit_col: str(other),
                    }
                )

        else:
            # Convert single string to list of strings.
            if isinstance(var_cols, str):
                var_cols = [var_cols]

            # Check that var_cols is valid.
            if not var_cols or any(c not in self._df for c in var_cols):
                raise Exception(
                    f"Argument var_cols must contain valid column "
                    f"name(s). Found: {var_cols}. Must be one of: "
                    f"{', '.join(self._df.columns)}."
                )

            # Determine locations where conversion needs to take place.
            if len(var_cols) == 1:
                locs = self._df[var_cols[0]].isin(list(other))
            else:
                locs = (
                    self._df[var_cols].apply(tuple, axis=1).isin(list(other))
                )

            # Define mappings for new units and conversion factors.
            mapping = (
                self._df.loc[locs, var_cols + [unit_col]]
                .drop_duplicates()
                .apply(
                    lambda row: pd.Series(
                        {
                            "index": tuple(row),
                            "conv_unit": other.get(
                                tuple(row.iloc[0:-1])
                                if len(row) > 2
                                else row.iloc[0],
                                row.iloc[-1],
                            ),
                            "conv_factor": getattr(
                                Quantity(row.iloc[-1]), subfunc
                            )(
                                other.get(
                                    tuple(row.iloc[0:-1])
                                    if len(row) > 2
                                    else row.iloc[0],
                                    row.iloc[-1],
                                ),
                                *contexts,
                                **ctx_kwargs,
                            ).m,
                        }
                    ),
                    axis=1,
                )
                .set_index("index")
            )
            conv_factors = mapping["conv_factor"].to_dict()
            conv_unit = mapping["conv_unit"].to_dict()

            # Apply mapping.
            mapping_key_col = self._df.loc[locs, var_cols + [unit_col]].apply(
                tuple, axis=1
            )
            new_values = self._df.loc[locs, value_col] * mapping_key_col.map(
                conv_factors
            )
            new_units = mapping_key_col.map(conv_unit)
            return self._df.assign(
                **{
                    value_col: self._df[value_col].where(~locs, new_values),
                    unit_col: self._df[unit_col].where(~locs, new_units),
                }
            )
