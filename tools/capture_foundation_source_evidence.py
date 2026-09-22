"""Capture bounded public permission-page fingerprints, never training content."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request
from urllib.parse import urlsplit

AI = Path(__file__).resolve().parents[1]
ROOT = AI / "data_governance/foundation"


def main():
    target = ROOT / "external-evidence.v1.json"
    if target.exists():
        raise ValueError("Evidence already captured; preserve it and use a new version")
    catalog = json.loads((ROOT / "proposed-sources.v1.json").read_text(encoding="utf-8"))
    rows = []
    for source in catalog["sources"]:
        url = source["licenseObservation"]["primaryUrl"]
        parts = urlsplit(url)
        if parts.scheme != "https" or parts.username or parts.password or parts.query:
            raise ValueError("Only public HTTPS permission references are allowed")
        row = {"sourceId": source["sourceId"], "url": url, "observedAtUtc": datetime.now(timezone.utc).isoformat(),
               "summarySha256": hashlib.sha256(source["licenseObservation"]["summary"].encode()).hexdigest(),
               "rawContentStored": False, "trainingUseAllowed": False, "corpusAcquired": False}
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "VoltForge-source-inventory/1.0 (permission-metadata-audit)"})
            with urllib.request.urlopen(request, timeout=20) as response:
                body = response.read(1_000_001)
                if len(body) > 1_000_000 or urlsplit(response.url).scheme != "https":
                    raise ValueError("Permission-page response exceeded its boundary")
                row.update(status="fingerprinted", responseSha256=hashlib.sha256(body).hexdigest(), bytes=len(body),
                           finalUrl=response.url, httpStatus=response.status)
        except Exception as error:
            row.update(status="response-unavailable", responseSha256=None, bytes=None, errorType=type(error).__name__)
        rows.append(row)
    result = {"schemaVersion": 1, "evidenceId": "vf-g2-permission-observations-v1", "networkAccessed": True,
              "scope": "Public permission metadata only; summaries were checked against primary publisher/maintainer pages. A response hash is not corpus approval or an archive of the page.", "records": rows}
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"report": str(target), "fingerprinted": sum(row["status"] == "fingerprinted" for row in rows), "total": len(rows), "corpusAcquired": False}))


if __name__ == "__main__":
    main()
