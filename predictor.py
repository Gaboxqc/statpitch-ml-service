# predictor.py  v3  --  StatPitch
# Improvements: H2H fix, rest days, goal momentum, xG features,
#               Dixon-Coles, calibration, stacking, neural network
import joblib, json
import numpy as np
from scipy.stats import poisson
from pathlib import Path
from datetime import datetime

BASE = Path(__file__).parent

# ─── Load models ─────────────────────────────────────────────────────────────
_xgb   = joblib.load(BASE / 'models' / 'model_xgb_v3.pkl')
_home  = joblib.load(BASE / 'models' / 'model_home_goals_v3.pkl')
_away  = joblib.load(BASE / 'models' / 'model_away_goals_v3.pkl')
_nn    = joblib.load(BASE / 'models' / 'model_nn_v3.pkl')
_sc    = joblib.load(BASE / 'models' / 'scaler_nn_v3.pkl')
_meta  = joblib.load(BASE / 'models' / 'model_meta_v3.pkl')

# ─── Load data ───────────────────────────────────────────────────────────────
with open(BASE / 'data' / 'model_config.json') as f:
    _cfg = json.load(f)
with open(BASE / 'data' / 'team_stats.json') as f:
    TEAM_STATS = json.load(f)
with open(BASE / 'data' / 'h2h_stats.json') as f:
    H2H_STATS = json.load(f)

FEATURE_COLS = _cfg['feature_cols']
ELO_DEFAULT  = float(_cfg.get('elo_start', 1500))
DC_RHO       = float(_cfg.get('dc_rho', -0.13))

_AVG = {
    'elo': ELO_DEFAULT,
    'gs_avg5': 1.30,     'gc_avg5': 1.30,     'pts_avg5': 1.20,    'win_rate5': 0.35,
    'gs_avg10': 1.30,    'gc_avg10': 1.30,    'pts_avg10': 1.20,   'win_rate10': 0.35,
    'goal_momentum': 1.30, 'current_rest_days': 14,
    'xg_avg5': 1.30,     'xga_avg5': 1.30,
    'xg_avg10': 1.30,    'xga_avg10': 1.30,
    'club_xg_overperf': 0.0,
    'squad_value_norm': 0.13, 'squad_age': 26.5,
}


def get_teams():
    return sorted(TEAM_STATS.keys())


def _rest_days(team, match_date=None):
    if match_date is None:
        match_date = datetime.now()
    elif isinstance(match_date, str):
        match_date = datetime.strptime(match_date, '%Y-%m-%d')
    last = TEAM_STATS.get(team, {}).get('last_match')
    if not last:
        return 14
    return int(min(max((match_date - datetime.strptime(last, '%Y-%m-%d')).days, 1), 30))


def _build_features(home, away, is_neutral, match_date=None):
    h   = TEAM_STATS.get(home, _AVG)
    a   = TEAM_STATS.get(away, _AVG)
    he  = h.get('elo', ELO_DEFAULT)
    ae  = a.get('elo', ELO_DEFAULT)
    h2h = H2H_STATS.get(f'{home}|{away}',
                          {'win_rate': 0.40, 'avg_goals': 2.50, 'num_games': 0})
    hr  = _rest_days(home, match_date)
    ar  = _rest_days(away, match_date)

    row = {
        'home_elo':              he,
        'away_elo':              ae,
        'elo_diff':              he - ae,
        'elo_prob_home':         1 / (1 + 10 ** ((ae - he) / 400)),
        'home_gs_avg5':          h.get('gs_avg5',    _AVG['gs_avg5']),
        'home_gc_avg5':          h.get('gc_avg5',    _AVG['gc_avg5']),
        'home_pts_avg5':         h.get('pts_avg5',   _AVG['pts_avg5']),
        'home_win_rate5':        h.get('win_rate5',  _AVG['win_rate5']),
        'home_gs_avg10':         h.get('gs_avg10',   _AVG['gs_avg10']),
        'home_gc_avg10':         h.get('gc_avg10',   _AVG['gc_avg10']),
        'home_pts_avg10':        h.get('pts_avg10',  _AVG['pts_avg10']),
        'home_win_rate10':       h.get('win_rate10', _AVG['win_rate10']),
        'away_gs_avg5':          a.get('gs_avg5',    _AVG['gs_avg5']),
        'away_gc_avg5':          a.get('gc_avg5',    _AVG['gc_avg5']),
        'away_pts_avg5':         a.get('pts_avg5',   _AVG['pts_avg5']),
        'away_win_rate5':        a.get('win_rate5',  _AVG['win_rate5']),
        'away_gs_avg10':         a.get('gs_avg10',   _AVG['gs_avg10']),
        'away_gc_avg10':         a.get('gc_avg10',   _AVG['gc_avg10']),
        'away_pts_avg10':        a.get('pts_avg10',  _AVG['pts_avg10']),
        'away_win_rate10':       a.get('win_rate10', _AVG['win_rate10']),
        'h2h_home_win_rate':     h2h['win_rate'],
        'h2h_avg_goals':         h2h['avg_goals'],
        'h2h_num_games':         h2h['num_games'],
        'is_neutral':            int(is_neutral),
        'tournament_weight':     1.00,
        'home_rest_days':        hr,
        'away_rest_days':        ar,
        'home_goal_momentum':    h.get('goal_momentum',    _AVG['goal_momentum']),
        'away_goal_momentum':    a.get('goal_momentum',    _AVG['goal_momentum']),
        'home_xg_avg5':          h.get('xg_avg5',          _AVG['xg_avg5']),
        'home_xga_avg5':         h.get('xga_avg5',         _AVG['xga_avg5']),
        'home_xg_avg10':         h.get('xg_avg10',         _AVG['xg_avg10']),
        'home_xga_avg10':        h.get('xga_avg10',        _AVG['xga_avg10']),
        'away_xg_avg5':          a.get('xg_avg5',          _AVG['xg_avg5']),
        'away_xga_avg5':         a.get('xga_avg5',         _AVG['xga_avg5']),
        'away_xg_avg10':         a.get('xg_avg10',         _AVG['xg_avg10']),
        'away_xga_avg10':        a.get('xga_avg10',        _AVG['xga_avg10']),
        'home_club_xg_overperf': h.get('club_xg_overperf', 0.0),
        'away_club_xg_overperf': a.get('club_xg_overperf', 0.0),
        'home_squad_value_norm': h.get('squad_value_norm', 0.13),
        'away_squad_value_norm': a.get('squad_value_norm', 0.13),
        'home_squad_age':        h.get('squad_age',        26.5),
        'away_squad_age':        a.get('squad_age',        26.5),
        'squad_value_diff':      h.get('squad_value_norm', 0.13)
                                  - a.get('squad_value_norm', 0.13),
    }
    return np.array([[row[c] for c in FEATURE_COLS]])


def _tau(x, y, lh, la, rho):
    if x == 0 and y == 0:   return 1 - lh * la * rho
    elif x == 0 and y == 1: return 1 + lh * rho
    elif x == 1 and y == 0: return 1 + la * rho
    elif x == 1 and y == 1: return 1 - rho
    else:                    return 1.0


def _dc_matrix(lh, la, max_g=9):
    mat = np.outer(
        [poisson.pmf(i, lh) for i in range(max_g)],
        [poisson.pmf(j, la) for j in range(max_g)],
    )
    for i in range(2):
        for j in range(2):
            mat[i, j] *= _tau(i, j, lh, la, DC_RHO)
    mat /= mat.sum()
    return mat


def _markets(lh, la, max_g=9):
    mat  = _dc_matrix(lh, la, max_g)
    idx  = np.array([[i + j for j in range(max_g)] for i in range(max_g)])
    btts = float((1 - poisson.pmf(0, lh)) * (1 - poisson.pmf(0, la)))
    top  = sorted(
        [{'score': f'{i}-{j}', 'probability': round(float(mat[i, j]), 4)}
         for i in range(max_g) for j in range(max_g)],
        key=lambda x: x['probability'], reverse=True,
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


def predict(home, away, is_neutral=True, match_date=None):
    feat   = _build_features(home, away, is_neutral, match_date)
    lh     = float(_home.predict(feat)[0])
    la     = float(_away.predict(feat)[0])

    xgb_p  = _xgb.predict_proba(feat)[0]
    poi_mat = _dc_matrix(lh, la)
    poi_p   = np.array([
        float(np.sum(np.triu(poi_mat, 1))),
        float(np.sum(np.diag(poi_mat))),
        float(np.sum(np.tril(poi_mat, -1))),
    ])
    nn_p    = _nn.predict_proba(_sc.transform(feat))[0]
    meta_x  = np.hstack([xgb_p, poi_p, nn_p]).reshape(1, -1)
    final_p = _meta.predict_proba(meta_x)[0]   # [away, draw, home]

    out = _markets(lh, la)
    out['match_result'] = {
        'home_win': round(float(final_p[2]), 4),
        'draw':     round(float(final_p[1]), 4),
        'away_win': round(float(final_p[0]), 4),
    }

    h   = TEAM_STATS.get(home, _AVG)
    a   = TEAM_STATS.get(away, _AVG)
    h2h = H2H_STATS.get(f'{home}|{away}',
                          {'win_rate': 0.40, 'avg_goals': 2.50, 'num_games': 0})
    return {
        'home_team':      home,
        'away_team':      away,
        'expected_goals': {'home': round(lh, 3), 'away': round(la, 3)},
        'team_info': {
            'home_elo':      round(h.get('elo', ELO_DEFAULT), 1),
            'away_elo':      round(a.get('elo', ELO_DEFAULT), 1),
            'elo_diff':      round(h.get('elo', ELO_DEFAULT) - a.get('elo', ELO_DEFAULT), 1),
            'h2h_games':     h2h['num_games'],
            'h2h_home_wins': round(h2h['win_rate'] * 100, 1),
            'home_squad_value_m': round(h.get('squad_value_eur', 0) / 1e6, 1),
            'away_squad_value_m': round(a.get('squad_value_eur', 0) / 1e6, 1),
        },
        'model_version': 'v3',
        **out,
    }
