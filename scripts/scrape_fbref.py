# scripts/scrape_fbref.py
# Run locally (not in Colab) — FBref blocks cloud/datacenter IPs aggressively.
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

try:
    import cloudscraper
    SCRAPER = cloudscraper.create_scraper()
    print('Using cloudscraper (better odds against bot detection).')
except ImportError:
    SCRAPER = requests
    print('cloudscraper not installed — falling back to requests. '
          'pip install cloudscraper if this gets blocked.')

ROOT = Path(__file__).parent.parent
OUT_PATH = ROOT / 'data' / 'fbref_xg_data.csv'

FBREF_URLS = [
    ('FIFA World Cup', 'https://fbref.com/en/comps/1/2022/schedule/2022-World-Cup-Scores-and-Fixtures'),
    ('FIFA World Cup', 'https://fbref.com/en/comps/1/2018/schedule/2018-World-Cup-Scores-and-Fixtures'),
    ('UEFA Euro', 'https://fbref.com/en/comps/676/2024/schedule/2024-UEFA-European-Championship-Scores-and-Fixtures'),
    ('UEFA Euro', 'https://fbref.com/en/comps/676/2020/schedule/2020-UEFA-European-Championship-Scores-and-Fixtures'),
    ('Copa America', 'https://fbref.com/en/comps/685/2024/schedule/2024-Copa-America-Scores-and-Fixtures'),
    ('Copa America', 'https://fbref.com/en/comps/685/2021/schedule/2021-Copa-America-Scores-and-Fixtures'),
    ('Africa Cup of Nations', 'https://fbref.com/en/comps/655/2023/schedule/2023-Africa-Cup-of-Nations-Scores-and-Fixtures'),
    ('Africa Cup of Nations', 'https://fbref.com/en/comps/655/2021/schedule/2021-Africa-Cup-of-Nations-Scores-and-Fixtures'),
    ('UEFA Nations League', 'https://fbref.com/en/comps/703/2024-2025/schedule/2024-2025-UEFA-Nations-League-Scores-and-Fixtures'),
    ('UEFA Nations League', 'https://fbref.com/en/comps/703/2022-2023/schedule/2022-2023-UEFA-Nations-League-Scores-and-Fixtures'),
]

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
}


def scrape_fbref(competition, url):
    try:
        resp = SCRAPER.get(url, headers=HEADERS, timeout=20)
        if resp.status_code == 429:
            print('    Blocked: rate limited (429)')
            return None
        if resp.status_code == 403:
            print('    Blocked: forbidden (403)')
            return None
        tables = pd.read_html(StringIO(resp.text))
        for t in tables:
            col_str = ' '.join(str(c) for c in t.columns)
            if 'xG' in col_str and ('Home' in col_str or 'Score' in col_str):
                return t
        print('    No xG table in response (likely a bot-detection page, not real content)')
        return None
    except Exception as e:
        print(f'    Error: {e}')
        return None


def parse_fbref_table(raw, competition):
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            ' '.join(str(x) for x in col if 'Unnamed' not in str(x)).strip() or str(col[-1])
            for col in df.columns
        ]
    cols = list(df.columns)
    xg_idx = [i for i, c in enumerate(cols) if str(c).strip() in ('xG', 'xg')]
    if len(xg_idx) < 2:
        xg_idx = [i for i, c in enumerate(cols) if 'xg' in str(c).lower() and 'expected' not in str(c).lower()]
    if len(xg_idx) < 2:
        return None
    df = df.rename(columns={cols[xg_idx[0]]: '_home_xg', cols[xg_idx[1]]: '_away_xg'})
    home_col = next((c for c in df.columns if str(c) == 'Home'), None)
    away_col = next((c for c in df.columns if str(c) == 'Away'), None)
    if home_col is None or away_col is None or 'Date' not in df.columns:
        return None
    df['_date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['_home_xg'] = pd.to_numeric(df['_home_xg'], errors='coerce')
    df['_away_xg'] = pd.to_numeric(df['_away_xg'], errors='coerce')
    result = df[['_date', home_col, away_col, '_home_xg', '_away_xg']].copy()
    result.columns = ['date', 'home_team', 'away_team', 'home_xg', 'away_xg']
    result['competition'] = competition
    result = result.dropna(subset=['date', 'home_xg', 'away_xg'])
    result = result[result['home_team'] != 'Home']
    return result


def main():
    chunks = []
    for competition, url in FBREF_URLS:
        print(f'Fetching: {competition}  ({url.split("/")[-1]})')
        raw = scrape_fbref(competition, url)
        if raw is not None:
            parsed = parse_fbref_table(raw, competition)
            if parsed is not None and len(parsed) > 0:
                chunks.append(parsed)
                print(f'  Got {len(parsed)} matches with xG')
            else:
                print('  Could not parse table structure')
        time.sleep(4)  # stay polite — don't lower this

    if not chunks:
        print('\nFAILED: 0 URLs returned usable data. FBref is still blocking this connection.')
        print('Next option: manually export each schedule page as CSV from the browser')
        print('(FBref has a "Share & Export" button on each page) and place them in data/fbref_manual/.')
        return

    xg_raw = pd.concat(chunks, ignore_index=True)
    xg_raw['date'] = pd.to_datetime(xg_raw['date'])
    xg_raw = xg_raw.drop_duplicates(subset=['date', 'home_team', 'away_team']).reset_index(drop=True)

    OUT_PATH.parent.mkdir(exist_ok=True)
    xg_raw.to_csv(OUT_PATH, index=False)
    print(f'\nSUCCESS: {len(xg_raw):,} xG records saved to {OUT_PATH}')
    print(xg_raw['competition'].value_counts().to_string())


if __name__ == '__main__':
    main()