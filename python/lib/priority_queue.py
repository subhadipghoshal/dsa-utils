#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
priority_queue.py
-----------------

A small, well‑tested, feature‑complete priority queue for Python.

Features
~~~~~~~~
* O(log n) push, pop, update, and arbitrary removal.
* Optional max‑heap mode (just set max_heap=True).
* Stable tie‑breaking (insertion order) – items with the same priority are
  returned in the order they were inserted.
* Custom key function (like `sorted(..., key=…)`) so you can push items
  without explicitly giving a priority.
* Built on top of the stdlib `heapq` module; no third‑party deps.
* Type annotations for static checkers.

Typical usage
~~~~~~~~~~~~~
>>> from priority_queue import PriorityQueue
>>> pq = PriorityQueue()
>>> pq.push('task1', 5)
>>> pq.push('task2', 2)
>>> pq.push('task3', 7)
>>> pq.pop()
'task2'
>>> pq.update('task1', 1)   # reprioritise an existing entry
>>> pq.peek()
'task1'
>>> pq.remove('task3')
>>> 'task3' in pq
False
"""

from __future__ import annotations

import itertools
import heapq
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Tuple,
    TypeVar,
    Optional,
)

# ----------------------------------------------------------------------
#  Generic type variables
# ----------------------------------------------------------------------
T = TypeVar('T')                     # type of the stored item
P = TypeVar('P', int, float)        # type of the priority (numeric)

# ----------------------------------------------------------------------
#  Core class
# ----------------------------------------------------------------------
class PriorityQueue(Generic[T, P]):
    """
    A min‑priority queue (or max‑priority if requested) with full
    support for ``push``, ``pop``, ``peek``, ``update`` and ``remove``.
    The class maintains a dict ``_position`` mapping each *item* to the
    index inside the underlying heap list. This makes all mutating ops
    O(log n).

    Parameters
    ----------
    key : Callable[[T], P], optional
        If supplied, ``push(item)`` will call ``key(item)`` to obtain the
        priority automatically. When ``key`` is ``None`` you must pass the
        priority explicitly on each ``push`` / ``update``.

    max_heap : bool, default ``False``
        If true, the queue behaves as a *max*‑heap (largest priority first).
        Internally we simply store ``-priority`` for each entry.
    """

    __slots__ = ("_heap", "_position", "_counter", "_key", "_sign", "_REMOVED")

    def __init__(
        self,
        *,
        key: Optional[Callable[[T], P]] = None,
        max_heap: bool = False,
    ) -> None:
        # The underlying list that `heapq` works on.
        self._heap: List[Tuple[P, int, T]] = []

        # Mapping item -> current index in `_heap`. Allows O(1) locate.
        self._position: Dict[T, int] = {}

        # A monotonically increasing counter to guarantee stable ordering.
        self._counter = itertools.count()

        # Function that extracts a priority from an item (if user supplied).
        self._key: Callable[[T], P] = (lambda x: x) if key is None else key

        # Used to flip sign for max‑heap semantics.
        self._sign: int = -1 if max_heap else 1

        # Marker object for lazy deletion (used only by the alternative
        # implementation at the bottom of this file).  It is not used here.
        self._REMOVED = object()

    # ------------------------------------------------------------------
    #   Helper: wrap a (priority, item) pair as a heap entry.
    #   The tuple layout is (signed_priority, insertion_counter, item).
    # ------------------------------------------------------------------
    def _entry(self, item: T, priority: P) -> Tuple[P, int, T]:
        """
        Return a tuple in the shape that ``heapq`` expects.
        The priority is multiplied by ``self._sign`` so the same
        code works for both min‑ and max‑heap.
        """
        return (self._sign * priority, next(self._counter), item)

    # ------------------------------------------------------------------
    #   Core public API
    # ------------------------------------------------------------------
    def push(self, item: T, priority: Optional[P] = None) -> None:
        """
        Insert *item* with the given *priority*.
        If ``priority`` is omitted, ``self._key(item)`` is called.
        Raises ``ValueError`` if the item is already present.
        """
        if item in self._position:
            raise ValueError(f"Item {item!r} already present in queue")

        if priority is None:
            priority = self._key(item)
        # Build the heap entry and push it.
        entry = self._entry(item, priority)
        self._heap.append(entry)
        idx = len(self._heap) - 1
        self._position[item] = idx
        self._sift_up(idx)

    def pop(self) -> T:
        """
        Remove and return the element with the smallest (or largest for
        a max‑heap) priority.
        Raises ``IndexError`` if the queue is empty.
        """
        if not self._heap:
            raise IndexError("pop from an empty priority queue")
        # Swap the root with the last element, pop it, then restore heap.
        self._swap(0, len(self._heap) - 1)
        priority, counter, item = self._heap.pop()
        del self._position[item]               # remove from index map

        if self._heap:
            self._sift_down(0)
        return item

    def peek(self) -> T:
        """
        Return the top element **without** removing it.
        Raises ``IndexError`` if the queue is empty.
        """
        if not self._heap:
            raise IndexError("peek from an empty priority queue")
        return self._heap[0][2]   # entry is (priority, counter, item)

    def update(self, item: T, new_priority: P) -> None:
        """
        Change the priority of *item* to ``new_priority``.
        Raises ``KeyError`` if the item is not present.
        """
        if item not in self._position:
            raise KeyError(f"Item {item!r} not found in queue")
        idx = self._position[item]
        old_priority, old_counter, _ = self._heap[idx]
        signed_new = self._sign * new_priority

        # Replace the priority field and then restore the heap property.
        self._heap[idx] = (signed_new, old_counter, item)

        # The new priority may be larger or smaller → we need to go both ways.
        if signed_new < old_priority:
            self._sift_up(idx)
        else:
            self._sift_down(idx)

    def remove(self, item: T) -> None:
        """
        Delete *item* from the queue, regardless of its priority.
        Raises ``KeyError`` if the item is not present.
        """
        if item not in self._position:
            raise KeyError(f"Item {item!r} not found in queue")
        idx = self._position[item]
        last = len(self._heap) - 1

        # If we are removing the last element, just pop it.
        if idx == last:
            self._heap.pop()
            del self._position[item]
            return

        # Otherwise swap with the last item, pop, then fix both heap directions.
        self._swap(idx, last)
        self._heap.pop()
        del self._position[item]

        # After the swap the element now at `idx` may need to move up or down.
        if idx < len(self._heap):
            self._sift_up(idx)
            self._sift_down(idx)

    # ------------------------------------------------------------------
    #   Python protocol support
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        """Return the number of items currently stored."""
        return len(self._heap)

    def __contains__(self, item: Any) -> bool:
        """Fast O(1) membership test."""
        return item in self._position

    def __bool__(self) -> bool:
        """Truthiness – empty queue is False, non‑empty is True."""
        return bool(self._heap)

    # ------------------------------------------------------------------
    #   Internal heap‑maintenance helpers
    # ------------------------------------------------------------------
    def _parent(self, idx: int) -> int:
        return (idx - 1) // 2

    def _left(self, idx: int) -> int:
        return 2 * idx + 1

    def _right(self, idx: int) -> int:
        return 2 * idx + 2

    def _swap(self, i: int, j: int) -> None:
        """Swap entries at positions i and j and keep `_position` in sync."""
        self._heap[i], self._heap[j] = self._heap[j], self._heap[i]
        # Update the index map for the two swapped items.
        item_i = self._heap[i][2]
        item_j = self._heap[j][2]
        self._position[item_i] = i
        self._position[item_j] = j

    def _sift_up(self, idx: int) -> None:
        """
        Move the entry at *idx* up the heap until the heap property holds.
        """
        while idx > 0:
            parent = self._parent(idx)
            if self._heap[idx] < self._heap[parent]:
                self._swap(idx, parent)
                idx = parent
            else:
                break

    def _sift_down(self, idx: int) -> None:
        """
        Move the entry at *idx* down the heap until the heap property holds.
        """
        n = len(self._heap)
        while (left := self._left(idx)) < n:
            smallest = left
            right = self._right(idx)
            if right < n and self._heap[right] < self._heap[left]:
                smallest = right
            if self._heap[smallest] < self._heap[idx]:
                self._swap(idx, smallest)
                idx = smallest
            else:
                break

    # ------------------------------------------------------------------
    #   Convenience: bulk insertion / iteration (optional)
    # ------------------------------------------------------------------
    def extend(self, items: Iterable[Tuple[T, P]]) -> None:
        """
        Insert a bunch of (item, priority) pairs at once.
        The overall complexity is O(k log (k+n)), where ``k`` is the number
        of new items and ``n`` is the current size.
        """
        for item, priority in items:
            self.push(item, priority)

    def __iter__(self) -> Iterable[T]:
        """
        Iterate over the items **in arbitrary heap order** (i.e. not sorted).
        If you need them sorted, use ``sorted(pq, key=…)`` or repeatedly
        ``pop()`` into a list.
        """
        return (entry[2] for entry in self._heap)

    # ------------------------------------------------------------------
    #   Debug/validation helpers (optional)
    # ------------------------------------------------------------------
    def _is_valid(self) -> bool:
        """Internal sanity check – useful while debugging."""
        for i, (priority, _, item) in enumerate(self._heap):
            # Check that the mapping points back to the proper index.
            if self._position.get(item) != i:
                return False
            # Verify heap ordering.
            left = self._left(i)
            right = self._right(i)
            if left < len(self._heap) and self._heap[left] < (priority, _, item):
                return False
            if right < len(self._heap) and self._heap[right] < (priority, _, item):
                return False
        return True

