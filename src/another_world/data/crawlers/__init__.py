"""Video / web crawlers.

Stage 1.2 of the roadmap implements:

- ``manifest.py``      :  Crawl manifest + approval workflow (here now).
- ``youtube.py``       :  yt-dlp wrapper with CC license filtering (TBD).
- ``vimeo.py``         :  Vimeo CC license fetcher (TBD).

Each crawl batch must be explicitly approved by the project owner before
execution (see :mod:`another_world.data.crawlers.manifest`).
"""

from another_world.data.crawlers.manifest import (
    CrawlApproval,
    CrawlConstraints,
    CrawlManifest,
    CrawlTarget,
    MAX_APPROVAL_AGE_SECONDS,
    gate_crawl,
)

__all__ = [
    "CrawlApproval",
    "CrawlConstraints",
    "CrawlManifest",
    "CrawlTarget",
    "MAX_APPROVAL_AGE_SECONDS",
    "gate_crawl",
]
