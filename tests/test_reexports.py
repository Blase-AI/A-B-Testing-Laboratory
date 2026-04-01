def test_core_reexport_modules_are_importable():
    # These modules are thin re-exports kept for backwards-compat / convenience.
    import app.ab_testing.plotting as plotting
    import app.ab_testing.power as power
    import app.ab_testing.validators as validators

    assert plotting.new_figure is not None
    assert power.perform_power_analysis is not None
    assert validators.validate_numeric_1d is not None
