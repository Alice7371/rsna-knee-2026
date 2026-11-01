"""Run the kaggle CLI with the www.googleapis.com -> storage.googleapis.com
rewrite applied to every request (this network cannot reach www.googleapis.com,
which makes uploads hang forever).

Usage: python kaggle_fix.py <kaggle args...>
"""
import sys

import requests

_orig = requests.Session.request


def patched(self, method, url, **kw):
    if isinstance(url, str) and "www.googleapis.com" in url:
        url = url.replace("www.googleapis.com", "storage.googleapis.com")
    kw.setdefault("timeout", 300)
    return _orig(self, method, url, **kw)


requests.Session.request = patched

from kaggle.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
