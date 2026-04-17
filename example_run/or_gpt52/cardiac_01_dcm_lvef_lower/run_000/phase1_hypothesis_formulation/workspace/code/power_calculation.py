# filename: power_calculation.py
from statsmodels.stats.power import TTestIndPower
analysis = TTestIndPower()
power = analysis.power(effect_size=0.5, nobs1=30, alpha=0.05, ratio=1)
print(f"Power at d=0.5: {power:.3f} ({'adequate' if power >= 0.8 else 'underpowered'})")