import numpy as np

import make_predictions


def test_preprosessing():
    features = [
        4.668102,
        193.681735,
        47580.991603,
        7.166639,
        359.948574,
        526.424171,
        13.894419,
        66.687695,
        4.435821,
    ]

    actual_preprocessed = make_predictions.preprocessing(features)
    expected_preprocessed = np.array(
        [
            -1.69398477e00,
            -5.66407442e-02,
            2.95701968e00,
            -1.21711995e-02,
            7.22710255e-01,
            1.23380913e00,
            -1.06104008e-01,
            1.45083389e-03,
            6.13677488e-01,
        ]
    )

    assert np.all(actual_preprocessed) == np.all(expected_preprocessed)


class ModelMock:
    """pseudo-model for testing"""

    def __init__(self, value):
        self.value = value

    def predict(self, X):
        n = len(X)
        return [self.value] * n


def test_predict():
    features = [
        4.668102,
        193.681735,
        47580.991603,
        7.166639,
        359.948574,
        526.424171,
        13.894419,
        66.687695,
        4.435821,
    ]
    model_mock = ModelMock(10.0)
    x = make_predictions.preprocessing(features)

    actual_prediction = make_predictions.predict(x)
    expected_prediction = False

    assert actual_prediction == expected_prediction
