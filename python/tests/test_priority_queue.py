#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_priority_queue.py
----------------------
A fairly exhaustive unit‑test suite for the `PriorityQueue` implementation
provided in `priority_queue.py`.

The tests cover:

* basic push / pop / peek semantics
* duplicate insertion, empty‑queue errors
* update (increase / decrease) and arbitrary removal
* membership, length, bool conversion
* max‑heap mode
* custom key function support
* stable tie‑breaking (insertion order for equal priorities)
* bulk insertion via `extend`
* iteration (the iterator returns exactly the stored items)
* internal heap‑invariant validation after random operations
* a “reference” test that compares the pop order with a fully sorted list
"""

import unittest
import random
import itertools
from typing import Any, List, Tuple

# ----------------------------------------------------------------------
# Adjust this import if you saved the implementation under another name.
# ----------------------------------------------------------------------
from priority_queue import PriorityQueue


class TestPriorityQueue(unittest.TestCase):

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _sorted_by_priority(pairs: List[Tuple[Any, int]]) -> List[Any]:
        """Return items sorted by priority (ascending)."""
        return [item for item, _ in sorted(pairs, key=lambda x: x[1])]

    # ------------------------------------------------------------------
    #  Basic functionality
    # ------------------------------------------------------------------
    def test_push_peek_pop_len_bool(self):
        pq = PriorityQueue[int, int]()
        items = [(5, 10), (2, 3), (7, 8), (1, 2)]  # (item, priority)

        # push one by one and check length/peek after each insertion
        for i, (itm, prio) in enumerate(items, start=1):
            pq.push(itm, prio)
            self.assertEqual(len(pq), i)
            self.assertTrue(pq)  # bool conversion = True while not empty

        # the smallest priority is 2 (item 1)
        self.assertEqual(pq.peek(), 1)

        # pop all items – they must come out in priority order
        expected_order = self._sorted_by_priority(items)
        popped = [pq.pop() for _ in range(len(pq))]
        self.assertEqual(popped, expected_order)

        # after everything is removed the queue is empty
        self.assertEqual(len(pq), 0)
        self.assertFalse(pq)

        # pop / peek on an empty queue raise the correct exception
        with self.assertRaises(IndexError):
            pq.pop()
        with self.assertRaises(IndexError):
            pq.peek()

    def test_duplicate_insert_raises(self):
        pq = PriorityQueue[str, int]()
        pq.push("a", 1)
        with self.assertRaises(ValueError):
            pq.push("a", 2)  # same item again

    # ------------------------------------------------------------------
    #  Update (decrease / increase) and arbitrary removal
    # ------------------------------------------------------------------
    def test_update_decrease_and_increase(self):
        pq = PriorityQueue[str, int]()
        pq.push("x", 30)
        pq.push("y", 20)
        pq.push("z", 40)

        # Decrease priority of 'z' → it should become the new top
        pq.update("z", 5)
        self.assertEqual(pq.peek(), "z")
        # Increase priority of 'x' → it should sink below 'y'
        pq.update("x", 50)
        self.assertEqual(pq.pop(), "z")
        self.assertEqual(pq.pop(), "y")
        self.assertEqual(pq.pop(), "x")

    def test_update_missing_raises(self):
        pq = PriorityQueue[int, int]()
        with self.assertRaises(KeyError):
            pq.update(99, 1)

    def test_remove(self):
        pq = PriorityQueue[int, int]()
        for i in range(5):
            pq.push(i, i * 10)

        # Remove a middle element
        pq.remove(2)
        self.assertNotIn(2, pq)
        self.assertEqual(len(pq), 4)
        self.assertTrue(all(item != 2 for item in pq))

        # Remove the last element (the one at the end of the internal list)
        last = max(pq._position.keys())
        pq.remove(last)
        self.assertNotIn(last, pq)

        # Remove a non‑existent element
        with self.assertRaises(KeyError):
            pq.remove(999)

    # ------------------------------------------------------------------
    #  Membership / containment, __len__, __bool__
    # ------------------------------------------------------------------
    def test_contains_and_len_bool(self):
        pq = PriorityQueue[int, int]()
        self.assertFalse(pq)  # empty → bool is False
        self.assertEqual(len(pq), 0)

        for i in range(3):
            pq.push(i, i)
        self.assertTrue(pq)  # non‑empty → True
        self.assertEqual(len(pq), 3)

        for i in range(3):
            self.assertIn(i, pq)
        self.assertNotIn(42, pq)

    # ------------------------------------------------------------------
    #  Max‑heap mode
    # ------------------------------------------------------------------
    def test_max_heap(self):
        maxpq = PriorityQueue[int, int](max_heap=True)
        data = [(10, 1), (5, 9), (7, 4), (2, 8)]
        for item, prio in data:
            maxpq.push(item, prio)

        # The largest priority (9) belongs to item 5
        self.assertEqual(maxpq.pop(), 5)
        # Next largest priority is 8 (item 2)
        self.assertEqual(maxpq.pop(), 2)
        # Followed by 4 and then 1
        self.assertEqual(maxpq.pop(), 7)
        self.assertEqual(maxpq.pop(), 10)

    # ------------------------------------------------------------------
    #  Custom key function + automatic priority extraction
    # ------------------------------------------------------------------
    def test_custom_key(self):
        # Simple class whose natural order is the integer stored in .value
        class Task:
            def __init__(self, name: str, value: int):
                self.name = name
                self.value = value

            def __repr__(self):
                return f"<Task {self.name}:{self.value}>"

            def __hash__(self):
                return hash(self.name)

            def __eq__(self, other):
                return isinstance(other, Task) and self.name == other.name

        # key extracts the numeric value from the task
        pq = PriorityQueue[Task, int](key=lambda t: t.value)

        tasks = [
            Task("A", 30),
            Task("B", 10),
            Task("C", 20),
        ]

        for t in tasks:
            pq.push(t)  # we *don’t* pass a priority explicitly

        # Because B has the smallest .value it should be the first popped
        self.assertEqual(pq.pop().name, "B")
        self.assertEqual(pq.pop().name, "C")
        self.assertEqual(pq.pop().name, "A")

    # ------------------------------------------------------------------
    #  Stable tie‑breaking (FIFO for equal priorities)
    # ------------------------------------------------------------------
    def test_stable_ordering(self):
        pq = PriorityQueue[str, int]()
        # Insert three items with the same priority (5)
        pq.push("first", 5)
        pq.push("second", 5)
        pq.push("third", 5)

        # The pop order must be the same as the insertion order
        self.assertEqual(pq.pop(), "first")
        self.assertEqual(pq.pop(), "second")
        self.assertEqual(pq.pop(), "third")

    # ------------------------------------------------------------------
    #  Bulk insertion via extend()
    # ------------------------------------------------------------------
    def test_extend_bulk(self):
        pq = PriorityQueue[int, int]()
        bulk = [(i, random.randint(1, 1000)) for i in range(50)]
        pq.extend(bulk)

        self.assertEqual(len(pq), 50)

        # Verify that every item we inserted is present
        for i, _ in bulk:
            self.assertIn(i, pq)

        # Verify pop order matches a fully sorted reference list
        reference = self._sorted_by_priority(bulk)
        popped = [pq.pop() for _ in range(len(pq))]
        self.assertEqual(popped, reference)

    # ------------------------------------------------------------------
    #  Iterator (should yield *all* items, order is irrelevant)
    # ------------------------------------------------------------------
    def test_iteration(self):
        pq = PriorityQueue[int, int]()
        data = [(10, 1), (20, 2), (30, 3)]
        for item, prio in data:
            pq.push(item, prio)

        iterated = set(pq)  # __iter__ returns the items
        expected = {10, 20, 30}
        self.assertEqual(iterated, expected)

    # ------------------------------------------------------------------
    #  Internal validation after random operations
    # ------------------------------------------------------------------
    def test_random_operations_and_internal_validation(self):
        random.seed(0)
        pq = PriorityQueue[int, int]()
        reference = {}  # dict item → priority (single copy)

        ops = 10_000
        items_range = range(0, 500)  # a pool of possible items

        for _ in range(ops):
            op_type = random.choices(
                ["push", "pop", "update", "remove"],
                weights=[0.4, 0.3, 0.2, 0.1],
                k=1,
            )[0]

            if (
                op_type == "push" or not reference
            ):  # need at least one item to pop/update/remove
                # Choose a fresh item (avoid duplicate)
                while True:
                    candidate = random.choice(items_range)
                    if candidate not in reference:
                        break
                prio = random.randint(0, 1000)
                pq.push(candidate, prio)
                reference[candidate] = prio

            elif op_type == "pop":
                # Pop from both structures and compare
                popped = pq.pop()
                # Find the smallest priority in the reference dict
                min_item = min(reference, key=reference.get)
                self.assertEqual(popped, min_item)
                del reference[min_item]

            elif op_type == "update":
                # Pick a random existing item and give it a new priority
                item = random.choice(list(reference.keys()))
                new_prio = random.randint(0, 1000)
                pq.update(item, new_prio)
                reference[item] = new_prio

            elif op_type == "remove":
                item = random.choice(list(reference.keys()))
                pq.remove(item)
                del reference[item]

            # After each operation, validate the heap's internal invariants.
            self.assertTrue(pq._is_valid())

        # Finally, empty the queue and make sure the remaining items match the sorted reference
        remaining = sorted(reference.items(), key=lambda kv: kv[1])
        popped_remaining = [(pq.pop(), None) for _ in range(len(pq))]
        popped_items = [itm for itm, _ in popped_remaining]
        self.assertEqual(popped_items, [itm for itm, _ in remaining])

    # ------------------------------------------------------------------
    #  Reference test: compare against a fully sorted list
    # ------------------------------------------------------------------
    def test_pop_matches_sorted_reference(self):
        random.seed(1)
        pq = PriorityQueue[int, int]()
        pairs = [(i, random.randint(0, 5000)) for i in range(200)]
        for item, prio in pairs:
            pq.push(item, prio)

        reference = [item for item, _ in sorted(pairs, key=lambda x: x[1])]
        popped = [pq.pop() for _ in range(len(pq))]
        self.assertEqual(popped, reference)


# ----------------------------------------------------------------------
# If you execute this file directly, run the tests.
# ----------------------------------------------------------------------
if __name__ == "__main__":
    unittest.main(verbosity=2)
