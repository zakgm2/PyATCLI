"""The parametric models and the curve-fit registry (models.py): each registered model has an initial-guess function
and parameter names that match its arity, and fitting data generated from it recovers the parameters it was built
from."""

import inspect
import unittest

import numpy as np
from numpy.testing import assert_allclose

import PyATCLI as pl
from PyATCLI import models
from PyATCLI.analysis.curve_fit import fit_model_to_segment

LINEAR = "Linear  (y = mx + b)"
DECAY = "Exponential Decay  (a*e^(-bx) + c)"
RISE = "Exponential Rise  (a*(1-e^(-bx)) + c)"
GAUSS = "Gaussian  (a*exp(-(x-u)^2/2s^2))"
SINE = "Sinusoidal  (a*sin(2*pi*f*x + phi) + c)"
DOUBLE = "Double Exponential  (a*e^(-bx) + c*e^(-dx) + k)"


class TestRegistryShape(unittest.TestCase):
    def test_it_offers_the_six_models_the_app_lists(self):
        self.assertEqual(list(pl.CURVE_FIT_MODELS), [LINEAR, DECAY, RISE, GAUSS, SINE, DOUBLE])
        self.assertIs(pl.CURVE_FIT_MODELS, models.CURVE_FIT_MODELS)

    def test_every_entry_is_a_model_a_guess_and_one_name_per_parameter(self):
        x = np.linspace(0, 10, 50)
        y = 3 * np.exp(-0.3 * x) + 1
        for name, (fn, p0_fn, pnames) in pl.CURVE_FIT_MODELS.items():
            with self.subTest(name):
                n_params = len(inspect.signature(fn).parameters) - 1                    # everything but x
                self.assertEqual(len(pnames), n_params)
                p0 = p0_fn(x, y)
                self.assertEqual(len(p0), n_params)
                self.assertTrue(np.all(np.isfinite(p0)))
                self.assertEqual(fn(x, *p0).shape, x.shape)

    def test_the_linear_guess_survives_a_zero_width_segment(self):
        p0 = pl.CURVE_FIT_MODELS[LINEAR][1](np.array([2.0, 2.0]), np.array([1.0, 5.0]))
        self.assertEqual(p0, [0, 1.0])


class TestFitsRecoverTheirParameters(unittest.TestCase):
    """Noise-free data from each model, fitted from the registry's own starting guess."""

    def fit(self, name, x, y):
        fn, p0_fn, _ = pl.CURVE_FIT_MODELS[name]
        res = fit_model_to_segment(x, y, fn, p0_fn)
        self.assertTrue(res["success"], res["error"])
        return res

    def test_linear(self):
        x = np.linspace(0, 20, 80)
        res = self.fit(LINEAR, x, 2.5 * x - 4.0)
        assert_allclose(res["popt"], [2.5, -4.0], rtol=1e-9, atol=1e-9)
        self.assertAlmostEqual(res["r2"], 1.0, places=9)

    def test_exponential_decay(self):
        x = np.linspace(0, 10, 100)
        res = self.fit(DECAY, x, models.single_exponential_model(x, 3.0, 0.4, 0.5))
        assert_allclose(res["popt"], [3.0, 0.4, 0.5], rtol=1e-5)

    def test_exponential_rise(self):
        x = np.linspace(0, 10, 100)
        res = self.fit(RISE, x, models.exponential_rise_model(x, 3.0, 0.4, 0.5))
        assert_allclose(res["popt"], [3.0, 0.4, 0.5], rtol=1e-5)

    def test_gaussian(self):
        x = np.linspace(0, 10, 200)
        res = self.fit(GAUSS, x, models.gaussian_model(x, 2.0, 5.0, 1.2))
        assert_allclose(res["popt"], [2.0, 5.0, 1.2], rtol=1e-5)

    def test_sinusoid_whose_frequency_is_near_the_one_hertz_starting_guess(self):
        """The registry starts every sinusoid fit at 1 Hz, so it finds a signal only if the phase it
        accumulates over the segment from that guess stays inside curve_fit's basin (here 1.01 Hz over
        5 s, 0.3 rad). A 1.05 Hz signal over 10 s (3 rad of drift) is not found from that start."""
        x = np.linspace(0, 5, 300)
        res = self.fit(SINE, x, models.sinusoidal_model(x, 1.5, 1.01, 0.3, 4.0))
        self.assertGreater(res["r2"], 0.999)
        self.assertAlmostEqual(res["popt"][1], 1.01, places=4)

    def test_double_exponential_on_a_bleaching_like_decay(self):
        x = np.linspace(0, 600, 600)
        y = models.double_exponential_model(x, 20.0, 0.04, 10.0, 0.0005, 5.0)
        res = self.fit(DOUBLE, x, y)
        self.assertGreater(res["r2"], 0.999)

    def test_float32_data_fits_the_same_as_float64(self):
        x = np.linspace(0, 10, 100)
        y = models.single_exponential_model(x, 3.0, 0.4, 0.5)
        a = self.fit(DECAY, x.astype(np.float32), y.astype(np.float32))
        b = self.fit(DECAY, x.astype(np.float32).astype(np.float64), y.astype(np.float32).astype(np.float64))
        np.testing.assert_array_equal(a["popt"], b["popt"])
        self.assertEqual(a["y_fit"].dtype, np.float64)


if __name__ == "__main__":
    unittest.main()
