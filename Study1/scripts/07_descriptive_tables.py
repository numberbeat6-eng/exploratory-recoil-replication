"""Export only the five tables presented in the specified manuscript."""
import importlib.util
import pandas as pd
from config import *


def main():
    spec = importlib.util.spec_from_file_location('daily', Path(__file__).with_name('02_daily_models.py'))
    daily = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(daily)
    panel = pd.read_csv(PANEL_DIR / 'selected_behavioral_panel.csv', parse_dates=['date'])
    panel = daily.add_calendar_value(panel, 'dose_sd', 1, 'lag_dose')
    sample = panel.dropna(subset=[*OUTCOMES, 'dose_sd', 'lag_dose', *CONTROLS])
    baseline = panel.loc[panel.post.eq(0)].groupby('user_id').preference_gap_is_click.mean()
    baseline = baseline.reindex(sample.user_id.unique())
    variables = {name: sample[name] for name in ['standard_n', 'opp_n']}
    variables['all_random_n_post'] = sample.loc[sample.post.eq(1), 'all_random_n']
    variables.update({name: sample[name] for name in ['dose_sd', *OUTCOMES]})
    variables['pretreatment_familiar_advantage'] = baseline
    rows = []
    for name, series in variables.items():
        series = series.dropna()
        rows.append(dict(variable=name, N=len(series), mean=series.mean(), sd=series.std(ddof=1),
                         p10=series.quantile(.1), median=series.median(), p90=series.quantile(.9)))
    pd.DataFrame(rows).to_csv(RESULT_DIR / 'table1_descriptive.csv', index=False)
    estimates = pd.read_csv(RESULT_DIR / 'behavioral_robustness_results.csv')
    effects = pd.read_csv(RESULT_DIR / 'behavioral_effect_sizes.csv')
    main_rows = estimates.loc[estimates.specification.eq('R4_distributed_lag')]
    main_rows.merge(effects, on='outcome', suffixes=('', '_effect')).to_csv(RESULT_DIR / 'table2_main.csv', index=False)
    estimates.loc[~estimates.specification.eq('R5_conditional_lead_placebo')].to_csv(RESULT_DIR / 'tableA1_robustness.csv', index=False)
    pd.read_csv(RESULT_DIR / 'mechanism_event_results.csv').to_csv(RESULT_DIR / 'table3_events.csv', index=False)
    hetero = pd.read_csv(RESULT_DIR / 'readi_style_heterogeneity_bootstrap.csv')
    hetero.loc[hetero.outcome.ne('preference_gap_is_click') & hetero.group.isin(['Q1', 'Q4', 'Q4-Q1'])].to_csv(RESULT_DIR / 'table4_heterogeneity.csv', index=False)
    print('Exported Tables 1-4 and A1; main sample:', len(sample), 'user-days,', sample.user_id.nunique(), 'users')


if __name__ == '__main__':
    main()
