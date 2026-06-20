# predictor.py  --  loads models and generates predictions for all markets
import joblib
import json
import numpy as np
from scipy.stats import poisson
from pathlib import Path

BASE = Path(__file__).parent

# ─── Load everything once at import time, not per request ────────────────────
_xgb  = joblib.load(BASE / 'models' / 'model_xgb_1x2.pkl')
_home = joblib.load(BASE / 'models' / 'model_home_goals.pkl')
_away = joblib.load(BASE / 'models' / 'model_away_goals.pkl')

with open(BASE / 'data' / 'model_config.json') as f:
    _cfg = json.load(f)

with open(BASE / 'data' / 'team_stats.json') as f:
    TEAM_STATS = json.load(f)

FEATURE_COLS = _cfg['feature_cols']
ELO_DEFAULT  = float(_cfg.get('elo_start', 1500))

# Default values used when a team is not in the dataset
_AVG = {
    'elo': ELO_DEFAULT,
    'gs_avg5': 1.30,  'gc_avg5': 1.30,  'pts_avg5': 1.20,  'win_rate5': 0.35,
    'gs_avg10': 1.30, 'gc_avg10': 1.30, 'pts_avg10': 1.20, 'win_rate10': 0.35,
}


def get_teams():
    return sorted(TEAM_STATS.keys())


def _build_features(home, away, is_neutral):
    h  = TEAM_STATS.get(home, _AVG)
    a  = TEAM_STATS.get(away, _AVG)
    he = h.get('elo', ELO_DEFAULT)
    ae = a.get('elo', ELO_DEFAULT)
    row = {
        'home_elo':        he,
        'away_elo':        ae,
        'elo_diff':        he - ae,
        'elo_prob_home':   1 / (1 + 10 ** ((ae - he) / 400)),
        'home_gs_avg5':    h.get('gs_avg5',    _AVG['gs_avg5']),
        'home_gc_avg5':    h.get('gc_avg5',    _AVG['gc_avg5']),
        'home_pts_avg5':   h.get('pts_avg5',   _AVG['pts_avg5']),
        'home_win_rate5':  h.get('win_rate5',  _AVG['win_rate5']),
        'home_gs_avg10':   h.get('gs_avg10',   _AVG['gs_avg10']),
        'home_gc_avg10':   h.get('gc_avg10',   _AVG['gc_avg10']),
        'home_pts_avg10':  h.get('pts_avg10',  _AVG['pts_avg10']),
        'home_win_rate10': h.get('win_rate10', _AVG['win_rate10']),
        'away_gs_avg5':    a.get('gs_avg5',    _AVG['gs_avg5']),
        'away_gc_avg5':    a.get('gc_avg5',    _AVG['gc_avg5']),
        'away_pts_avg5':   a.get('pts_avg5',   _AVG['pts_avg5']),
        'away_win_rate5':  a.get('win_rate5',  _AVG['win_rate5']),
        'away_gs_avg10':   a.get('gs_avg10',   _AVG['gs_avg10']),
        'away_gc_avg10':   a.get('gc_avg10',   _AVG['gc_avg10']),
        'away_pts_avg10':  a.get('pts_avg10',  _AVG['pts_avg10']),
        'away_win_rate10': a.get('win_rate10', _AVG['win_rate10']),
        'h2h_home_win_rate': 0.40,
        'h2h_avg_goals':     2.50,
        'h2h_num_games':     0,
        'is_neutral':        int(is_neutral),
        'tournament_weight': 1.00,
    }
    return np.array([[row[c] for c in FEATURE_COLS]])


def _markets(lh, la, max_g=9):
    mat = np.outer(
        [poisson.pmf(i, lh) for i in range(max_g)],
        [poisson.pmf(j, la) for j in range(max_g)],
    )
    idx  = np.array([[i + j for j in range(max_g)] for i in range(max_g)])
    btts = float((1 - poisson.pmf(0, lh)) * (1 - poisson.pmf(0, la)))
    top  = sorted(
        [{'score': f'{i}-{j}', 'probability': round(float(mat[i, j]), 4)}
         for i in range(max_g) for j in range(max_g)],
        key=lambda x: x['probability'],
        reverse=True,
    )[:10]
    return {
        'match_result': {
            'home_win': round(float(np.sum(np.tril(mat, -1))), 4),
            'draw':     round(float(np.sum(np.diag(mat))),     4),
            'away_win': round(float(np.sum(np.triu(mat, 1))),  4),
        },
        'over_under': {
            'over_1_5': round(float(np.sum(mat[idx > 1])), 4),
            'over_2_5': round(float(np.sum(mat[idx > 2])), 4),
            'over_3_5': round(float(np.sum(mat[idx > 3])), 4),
        },
        'btts': {'yes': round(btts, 4), 'no': round(1 - btts, 4)},
        'correct_score': top,
    }


def predict(home, away, is_neutral=True):
    feat = _build_features(home, away, is_neutral)
    lh = float(_home.predict(feat)[0])
    la = float(_away.predict(feat)[0])
    xgb_p = _xgb.predict_proba(feat)[0]  # [away_prob, draw_prob, home_prob]
    out = _markets(lh, la)
    mr = out['match_result']

    # Blend XGBoost (60%) + Poisson (40%) for 1X2
    out['match_result'] = {
        'home_win': round(float(0.6 * xgb_p[2] + 0.4 * mr['home_win']), 4),
        'draw': round(float(0.6 * xgb_p[1] + 0.4 * mr['draw']), 4),
        'away_win': round(float(0.6 * xgb_p[0] + 0.4 * mr['away_win']), 4),
    }

    h = TEAM_STATS.get(home, _AVG)
    a = TEAM_STATS.get(away, _AVG)

    
    he_val = h.get('elo', ELO_DEFAULT)
    ae_val = a.get('elo', ELO_DEFAULT)


    if np.isnan(he_val): he_val = ELO_DEFAULT
    if np.isnan(ae_val): ae_val = ELO_DEFAULT
    # -----------------------

    return {
        'home_team': home,
        'away_team': away,
        'expected_goals': {'home': round(lh, 3), 'away': round(la, 3)},
        'team_info': {
            'home_elo': round(he_val, 1),
            'away_elo': round(ae_val, 1),
            'elo_diff': round(he_val - ae_val, 1),
        },
        **out,
    }
