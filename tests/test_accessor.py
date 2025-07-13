import unittest

import numpy as np
import pandas as pd

from team import TEAMAccessor, CalcVariable, U


class TestAccessor(unittest.TestCase):
    def test_sum_over(self):
        # Define test dataframe.
        df = pd.DataFrame.from_dict({
            'region': 3 * 3 * ['USA'] + 3 * 3 * ['Russia'] + 3 * 3 * ['Algeria'],
            'period': 3 * (3 * [2030] + 3 * [2040] + 3 * [2050]),
            'fuel': 3 * 3 * ['coal', 'oil', 'gas'],
            'unit': 3 * 3 * ['tce', 'EJ', 'TWh'],
            'value': np.random.rand(3 * 3 * 3),
        })

        # Sum over column where units are always the same within field.
        df_new = df.team.sum_over('region')
        assert len(df_new) == 9
        assert len(df_new.columns) == len(df.columns) - 1
        assert 'region' not in df_new

        # Sum over column with different units.
        df_new = df.team.sum_over('fuel')
        assert len(df_new) == 3 * 3
        assert len(df_new.columns) == len(df.columns) - 1
        assert 'fuel' not in df_new
        assert df_new['unit'].nunique() == 1

    def test_unit_conversion(self):
        # Define test dataframe.
        df = pd.DataFrame.from_dict({
            'region': 3 * 3 * ['USA'] + 3 * 3 * ['Russia'] + 3 * 3 * ['Algeria'],
            'period': 3 * (3 * [2030] + 3 * [2040] + 3 * [2050]),
            'fuel': 3 * 3 * ['coal', 'oil', 'gas'],
            'unit': 3 * 3 * ['tce', 'EJ', 'TWh'],
            'value': np.random.rand(3 * 3 * 3),
        })

        # Convert all units to TWh.
        df_new = df.team.unit_to('TWh')
        assert (df_new['unit'] == 'TWh').all()

        # Convert only coal to TWh.
        df_new = df.team.unit_to({'coal': 'TWh'}, var_cols='fuel')
        assert not (df_new['unit'] == 'TWh').all()
        assert (df_new.loc[df['fuel']=='coal', 'unit'] == 'TWh').all()
        assert (df.loc[df['fuel']!='coal'] == df_new.loc[df['fuel']!='coal']).all().all()

        # Convert only coal in one region to TWh.
        df_new = df.team.unit_to({('coal', 'USA'): 'TWh'}, var_cols=['fuel', 'region'])
        assert not (df_new['unit'] == 'TWh').all()
        assert (df_new.loc[(df['fuel']=='coal') * (df['region']=='USA'), 'unit'] == 'TWh').all()
        assert (df.loc[df['fuel']!='coal'] == df_new.loc[df['fuel']!='coal']).all().all()

        # Convert to "preferred" units.
        df_new = df.team.unit_to_preferred([U('TWh')])
        assert (df_new['unit'] == 'TWh').all()

        # Define test dataframe.
        df = pd.DataFrame.from_dict({
            'region': 3 * 3 * ['USA'] + 3 * 3 * ['Russia'] + 3 * 3 * ['Algeria'],
            'period': 3 * (3 * [2030] + 3 * [2040] + 3 * [2050]),
            'fuel': 3 * 3 * ['coal', 'oil', 'gas'],
            'unit': 3 * 3 * ['kt a_CH4', 'kt a_CH4', 'kt a_CH4'],
            'value': np.random.rand(3 * 3 * 3),
        })

        # Convert assuming a global-warming potential.
        df_new = df.team.unit_to('Mt a_CO2eq', 'AR6GWP100')

    def test_perform(self):
        # Define test dataframe.
        df1 = pd.DataFrame.from_dict({
            'region': 3 * 3 * ['USA'] + 3 * 3 * ['Russia'] + 3 * 3 * ['Algeria'],
            'period': 3 * (3 * [2030] + 3 * [2040] + 3 * [2050]),
            'fuel': 3 * 3 * ['coal', 'oil', 'gas'],
            'variable': 3 * 3 * 3 * ['quantity'],
            'unit': 3 * 3 * ['tce', 'EJ', 'TWh'],
            'value': np.random.rand(3 * 3 * 3),
        })
        df2 = pd.DataFrame.from_dict({
            'fuel': ['coal', 'oil', 'gas'],
            'variable': 3 * ['price'],
            'unit': 3 * ['EUR/MWh'],
            'value': np.random.rand(3),
        })
        df = pd.concat([df1, df2])

        # Calculate new variable: prices times quantity.
        df_new = (
            df
            .team.perform(CalcVariable('total=price*quantity'), only_new=True)
            .team.unit_to('million EUR')
        )
        assert len(df_new) == 3 * 3 * 3
        assert len(df_new.columns) == len(df.columns)
        assert df_new['variable'].nunique() == 1


if __name__ == '__main__':
    unittest.main()
