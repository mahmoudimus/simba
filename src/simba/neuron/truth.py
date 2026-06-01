"""Retired Truth DB module.

The ``truth_add``/``truth_query`` API and the ``proven_facts`` table have been
superseded by the temporal knowledge graph (``simba.kg.store``).  Facts are now
recorded via ``kg_add`` and queried via ``kg_query``; the legacy
``proven_facts`` table is backed up and dropped on first connect by
``simba.kg.store.backup_and_drop_proven_facts``.  This module is kept as an
empty placeholder so stale imports resolve without re-creating the old schema.
"""

from __future__ import annotations
