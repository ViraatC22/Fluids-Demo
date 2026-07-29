import math
import unittest

from streamlit.testing.v1 import AppTest

import app


class PhysicsTests(unittest.TestCase):
    def test_circle_area_from_diameter(self) -> None:
        self.assertAlmostEqual(
            app.circle_area_from_diameter(0.10),
            math.pi * 0.05**2,
        )

    def test_constriction_preserves_flow_and_reduces_pressure(self) -> None:
        (
            area_1,
            area_2,
            velocity_1,
            velocity_2,
            mass_flow,
            pressure_2,
            pressure_change,
            residual,
            flow,
        ) = app.compute_flow_and_pressures(
            rho=1000.0,
            Q_m3s=0.01,
            d1_m=0.06,
            d2_m=0.03,
            p1_pa=200_000.0,
            dh_m=0.0,
            g=9.81,
        )

        self.assertAlmostEqual(area_1 * velocity_1, flow)
        self.assertAlmostEqual(area_2 * velocity_2, flow)
        self.assertAlmostEqual(velocity_2 / velocity_1, 4.0)
        self.assertAlmostEqual(mass_flow, 10.0)
        self.assertLess(pressure_2, 200_000.0)
        self.assertAlmostEqual(pressure_change, pressure_2 - 200_000.0)
        self.assertAlmostEqual(residual, 0.0)

    def test_elevation_change_applies_hydrostatic_pressure_drop(self) -> None:
        result = app.compute_flow_and_pressures(
            rho=1000.0,
            Q_m3s=0.01,
            d1_m=0.06,
            d2_m=0.06,
            p1_pa=200_000.0,
            dh_m=2.0,
            g=9.81,
        )

        self.assertAlmostEqual(result[5], 200_000.0 - 1000.0 * 9.81 * 2.0)
        self.assertAlmostEqual(result[6], -19_620.0)

    def test_zero_flow_is_a_valid_stationary_case(self) -> None:
        result = app.compute_flow_and_pressures(
            rho=1000.0,
            Q_m3s=0.0,
            d1_m=0.06,
            d2_m=0.03,
            p1_pa=200_000.0,
            dh_m=0.0,
            g=9.81,
        )

        self.assertEqual(result[2], 0.0)
        self.assertEqual(result[3], 0.0)
        self.assertEqual(result[4], 0.0)
        self.assertEqual(result[5], 200_000.0)

    def test_rejects_nonphysical_inputs(self) -> None:
        valid = {
            "rho": 1000.0,
            "Q_m3s": 0.01,
            "d1_m": 0.06,
            "d2_m": 0.03,
            "p1_pa": 200_000.0,
            "dh_m": 0.0,
            "g": 9.81,
        }
        invalid_cases = (
            {"rho": 0.0},
            {"Q_m3s": -0.01},
            {"d1_m": 0.0},
            {"d2_m": -0.03},
            {"g": -9.81},
            {"p1_pa": math.inf},
        )

        for override in invalid_cases:
            with self.subTest(override=override), self.assertRaises(ValueError):
                app.compute_flow_and_pressures(**(valid | override))


class StreamlitSmokeTests(unittest.TestCase):
    def test_app_renders_primary_workflow_without_exceptions(self) -> None:
        app_test = AppTest.from_file("app.py", default_timeout=10)
        app_test.run()

        self.assertEqual(len(app_test.exception), 0)
        self.assertEqual(
            [tab.label for tab in app_test.tabs],
            ["Simulation", "Data Log", "Explanation"],
        )
        self.assertEqual(len(app_test.slider), 8)
        self.assertEqual(len(app_test.toggle), 3)

    def test_trial_can_be_recorded(self) -> None:
        app_test = AppTest.from_file("app.py", default_timeout=10)
        app_test.run()
        app_test.button[0].click().run()

        self.assertEqual(len(app_test.exception), 0)
        self.assertEqual(len(app_test.dataframe), 1)
        self.assertEqual(app_test.dataframe[0].value.shape, (1, 19))
        self.assertEqual(app_test.dataframe[0].value.iloc[0]["Trial"], 1)


if __name__ == "__main__":
    unittest.main()
