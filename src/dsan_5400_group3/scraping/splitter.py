"""
splitter.py

Provides helper functions for evenly splitting lists of page IDs across
multiple workers to support parallel scraping.
"""

# src/dsan_5400_group3/data/splitter.py

def split_into_chunks(list_of_items, n_workers):
    """Split list into n roughly equal chunks."""
    k, m = divmod(len(list_of_items), n_workers)
    return [
        list_of_items[i*k + min(i, m):(i+1)*k + min(i+1, m)]
        for i in range(n_workers)
    ]
