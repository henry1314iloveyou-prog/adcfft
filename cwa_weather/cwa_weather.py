#!/usr/bin/env python3
"""從中央氣象署開放資料平台抓取「一般天氣預報-今明36小時天氣預報」(F-C0032-001)。

用法:
    export CWA_API_KEY=CWA-xxxxxxxx      # 到 https://opendata.cwa.gov.tw/ 申請
    python cwa_weather.py                 # 抓一次
    python cwa_weather.py --daemon        # 常駐，每天 08:00 抓一次
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests

URL = "https://opendata.cwa.gov.tw/api/v1/rest/datastore/F-C0032-001"
NAMES = {"Wx": "天氣現象", "PoP": "降雨機率(%)", "MinT": "最低溫(°C)",
         "CI": "舒適度", "MaxT": "最高溫(°C)"}


def fetch(api_key, locations=None):
    params = {"Authorization": api_key, "format": "JSON"}
    if locations:
        params["locationName"] = ",".join(locations)
    r = requests.get(URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    if data.get("success") not in ("true", True):
        raise RuntimeError(f"API 回傳失敗: {data}")
    return data


def summarize(data):
    """轉成易讀文字。"""
    lines = []
    for loc in data["records"]["location"]:
        lines.append(f"【{loc['locationName']}】")
        elems = {e["elementName"]: e["time"] for e in loc["weatherElement"]}
        for i, t in enumerate(elems["Wx"]):
            lines.append(f"  {t['startTime']} ~ {t['endTime']}")
            for key, label in NAMES.items():
                if key in elems:
                    lines.append(f"    {label}: {elems[key][i]['parameter']['parameterName']}")
    return "\n".join(lines)


def run_once(api_key, locations, outdir):
    data = fetch(api_key, locations)
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    (outdir / f"weather_{stamp}.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    text = summarize(data)
    print(f"[{datetime.now():%F %T}] 已儲存 weather_{stamp}.json\n{text}", flush=True)


def seconds_until(hour, minute=0):
    now = datetime.now()
    nxt = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if nxt <= now:
        nxt += timedelta(days=1)
    return (nxt - now).total_seconds()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--daemon", action="store_true", help="常駐並於每天指定時間執行")
    p.add_argument("--hour", type=int, default=8, help="每天執行的小時 (預設 8)")
    p.add_argument("--location", action="append", help="縣市名稱，如 臺北市，可重複；預設全部")
    p.add_argument("--outdir", type=Path, default=Path("data"))
    a = p.parse_args()

    key = os.environ.get("CWA_API_KEY")
    if not key:
        sys.exit("請先設定環境變數 CWA_API_KEY")

    if not a.daemon:
        run_once(key, a.location, a.outdir)
        return
    while True:
        time.sleep(seconds_until(a.hour))
        try:
            run_once(key, a.location, a.outdir)
        except Exception as e:  # 失敗不要讓常駐程式結束
            print(f"[{datetime.now():%F %T}] 錯誤: {e}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
