from .base import BaseABTest, TwoGroupContinuousABTest
from .methods import BayesianABTest, BootstrapABTest, ClassicABTest
from .methods import SequentialSPRT, SPRTResult, NormalSPRTSimulator

from .core import (
    DEFAULT_STYLE,
    PlotStyle,
    new_figure,
    new_figure_grid,
    perform_power_analysis,
    validate_numeric_1d,
)

from .methods.classic import plots as classic_plots, stats as classic_stats
from .methods.bootstrap import plots as bootstrap_plots, stats as bootstrap_stats
from .methods.bayesian import plots as bayesian_plots, stats as bayesian_stats
from .methods.sequential import plot_sprt_history

ABTest = ClassicABTest

__all__ = [
    "BaseABTest",
    "TwoGroupContinuousABTest",
    "ClassicABTest",
    "BootstrapABTest",
    "BayesianABTest",
    "ABTest",
    "SequentialSPRT",
    "SPRTResult",
    "NormalSPRTSimulator",
    "plot_sprt_history",
    "classic_stats",
    "classic_plots",
    "bootstrap_stats",
    "bootstrap_plots",
    "bayesian_stats",
    "bayesian_plots",
    "validate_numeric_1d",
    "perform_power_analysis",
    "PlotStyle",
    "DEFAULT_STYLE",
    "new_figure",
    "new_figure_grid",
]