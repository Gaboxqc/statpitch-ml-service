# main.py  --  FastAPI application entry point
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import predictor

app = FastAPI(
    title='World Cup Predictor API',
    description='Predict match outcomes for international football using ML',
    version='1.0.0',
)

# CORS: allows any frontend (React, HTML) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)


class PredictRequest(BaseModel):
    home_team:  str
    away_team:  str
    is_neutral: Optional[bool] = True


def _validate(home: str, away: str):
    teams = predictor.get_teams()
    if home not in teams:
        raise HTTPException(status_code=404, detail=f'Team not found: {home}')
    if away not in teams:
        raise HTTPException(status_code=404, detail=f'Team not found: {away}')
    if home == away:
        raise HTTPException(status_code=400, detail='Teams must be different')


@app.get('/')
def root():
    return {
        'service':   'World Cup Predictor API',
        'version':   '1.0.0',
        'endpoints': ['/health', '/teams', '/predict', '/docs'],
    }


@app.get('/health')
def health():
    return {
        'status':          'ok',
        'teams_available': len(predictor.TEAM_STATS),
    }


@app.get('/teams')
def get_teams():
    ranked = sorted(
        [{'team': k, 'elo': round(v.get('elo', 1500), 1)}
         for k, v in predictor.TEAM_STATS.items()],
        key=lambda x: x['elo'],
        reverse=True,
    )
    return {'count': len(ranked), 'teams': ranked}


@app.post('/predict')
def predict_post(req: PredictRequest):
    _validate(req.home_team, req.away_team)
    return predictor.predict(req.home_team, req.away_team, req.is_neutral)


@app.get('/predict/{home_team}/{away_team}')
def predict_get(home_team: str, away_team: str, is_neutral: bool = True):
    _validate(home_team, away_team)
    return predictor.predict(home_team, away_team, is_neutral)
