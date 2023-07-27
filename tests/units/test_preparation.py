import unittest
import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import GridSearchCV

from src.data.preparation import OutlierHandler

class TestOutlierHandler(unittest.TestCase):
    def setUp(self):
        # Generate a sample dataset for testing
        X, y = make_regression(n_samples=100, n_features=3, random_state=42)
        self.data = pd.DataFrame(X, columns=['feature1', 'feature2', 'feature3'])
        self.target = pd.Series(y, name='target')

    def test_transform_with_transformer(self):
        # Test the transformation using a transformer
        handler = OutlierHandler(strategy='transformer', transformer='log')
        transformed_data = handler.transform(self.data)
        np.testing.assert_array_almost_equal(np.log(self.data), transformed_data)

    def test_transform_with_statistics_median(self):
        # Test the transformation using the median strategy
        handler = OutlierHandler(strategy='statistics', statistics='median')
        transformed_data = handler.transform(self.data)
        median_values = self.data.median()
        expected_data = self.data.apply(lambda x: np.where(x < median_values[x.name], median_values[x.name], x))
        pd.testing.assert_frame_equal(expected_data, transformed_data)

    def test_transform_with_statistics_trimmed_mean(self):
        # Test the transformation using the trimmed mean strategy
        handler = OutlierHandler(strategy='statistics', statistics='trimmed_mean')
        transformed_data = handler.transform(self.data)
        trimmed_mean_values = self.data.mean().clip(lower=self.data.quantile(0.05), upper=self.data.quantile(0.95))
        expected_data = self.data.apply(lambda x: np.where(x < trimmed_mean_values[x.name], trimmed_mean_values[x.name], x))
        pd.testing.assert_frame_equal(expected_data, transformed_data)

    def test_transform_with_statistics_winsorization(self):
        # Test the transformation using the winsorization strategy
        handler = OutlierHandler(strategy='statistics', statistics='winsorization')
        transformed_data = handler.transform(self.data)
        expected_data = pd.DataFrame(index=self.data.index)
        for col in self.data.columns:
            expected_data[col] = handler._apply_winsorization(self.data[col])
        pd.testing.assert_frame_equal(expected_data, transformed_data)

    # def test_transform_with_random_forest(self):
    #     # Test the transformation using the random forest strategy
    #     handler = OutlierHandler(strategy='random_forest')

    #     transformed_data = handler.transform(self.data)
    #     regressor = RandomForestRegressor()
    #     transformer_regressor = TransformedTargetRegressor(regressor=regressor, transformer=FunctionTransformer())
    #     param_grid = {
    #         'n_estimators': [100, 200, 500],
    #         'max_depth': [None, 10, 20],
    #         'min_samples_split': [2, 5, 10]
    #     }
    #     grid_search = GridSearchCV(transformer_regressor, param_grid=param_grid, cv=5)
    #     grid_search.fit(self.data, self.data)
    #     expected_data = pd.DataFrame(grid_search.predict(self.data), index=self.data.index, columns=self.data.columns)
    #     pd.testing.assert_frame_equal(expected_data, transformed_data)

    def test_transform_with_zscore(self):
        # Test the transformation using the z-score strategy
        handler = OutlierHandler(strategy='zscore')
        transformed_data = handler.transform(self.data)
        z_scores = (self.data - self.data.mean()) / self.data.std()
        expected_data = self.data.mask(np.abs(z_scores) > 3)
        pd.testing.assert_frame_equal(expected_data, transformed_data)

    def test_transform_with_remove(self):
        # Test the transformation using the remove strategy
        handler = OutlierHandler(strategy='remove')
        transformed_data = handler.transform(self.data)
        expected_data = pd.DataFrame(index=self.data.index)
        for col in self.data.columns:
            series = self.data[col]
            series_mean = series.mean()
            series_std = series.std()
            expected_data[col] = series.mask(np.abs((series - series_mean) / series_std) > 3)
        pd.testing.assert_frame_equal(expected_data, transformed_data)

if __name__ == '__main__':
    unittest.main()
