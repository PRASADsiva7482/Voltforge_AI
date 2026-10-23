"""Opt-in live checks using only fixed public Arduino documentation/query.

Prints status/counts, never keys, request headers, queries or source text.
No database writes or model invocation. Exit 2 means live search is unverified.
"""
import argparse
import asyncio
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from api.research import ResearchRequest
from config import get_settings
from engine.web_research import PublicHTTP, ResearchError, WebResearch, extract_text


async def verify(live_search):
    settings = get_settings()
    try:
        async with PublicHTTP(settings) as http:
            page = await http.get('https://docs.arduino.cc/hardware/uno-rev3/')
        text = extract_text(page.body.decode('utf-8', errors='replace'))
        if not text:
            print('FAIL public-page extraction: no readable text')
            return 1
        print(f'PASS public HTTPS page: {len(page.body)} bytes; {len(text)} excerpt characters')
    except ResearchError as error:
        print(f'FAIL public-page retrieval: {error.code}')
        return 1
    if not live_search:
        print('NOT VERIFIED: provider search (use --live-search after configuring credentials)')
        return 2
    try:
        result = await WebResearch().search(ResearchRequest(
            mode='on', publicQuery='Arduino Uno R3 technical specifications site:docs.arduino.cc',
            consentToExternalQuery=True,
        ))
    except ResearchError as error:
        print(f'NOT VERIFIED: provider search ({error.code})')
        return 2
    pages = sum(s['kind'] == 'web_page' for s in result['sources'])
    print(f'Provider search: {result["status"]}; {len(result["sources"])} sources; {pages} fetched pages')
    return 0 if result['sources'] and pages else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--live-search', action='store_true', help='Send a fixed public Arduino query to configured Brave API')
    args = parser.parse_args()
    raise SystemExit(asyncio.run(verify(args.live_search)))
