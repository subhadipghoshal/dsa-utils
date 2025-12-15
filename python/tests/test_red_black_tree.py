#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_red_black_tree.py
---------------------

Exercises the RedBlackTree implementation with a fairly exhaustive set of
tests covering:

* basic CRUD (insert, lookup, delete)
* duplicate‑key replacement
* ordered traversal and iterator behaviour
* min / max / predecessor / successor
* randomised bulk insert/delete compared against Python's built‑in dict
* validation of red‑black invariants after each operation
"""

import random
import unittest
from typing import Tuple, List

from red_black_tree import RedBlackTree, RED, BLACK


class TestRedBlackTree(unittest.TestCase):
    # ------------------------------------------------------------------
    #  Basic CRUD
    # ------------------------------------------------------------------
    def test_insert_and_lookup(self):
        rbt = RedBlackTree[int, str]()
        rbt[10] = "ten"
        rbt[5] = "five"
        rbt[20] = "twenty"

        self.assertEqual(rbt[10], "ten")
        self.assertEqual(rbt[5], "five")
        self.assertEqual(rbt[20], "twenty")
        self.assertEqual(len(rbt), 3)

    def test_duplicate_key_replaces_value(self):
        rbt = RedBlackTree[int, str]()
        rbt[1] = "a"
        rbt[1] = "b"  # replace
        self.assertEqual(rbt[1], "b")
        self.assertEqual(len(rbt), 1)

    def test_delete(self):
        rbt = RedBlackTree[int, str]()
        for k in range(5):
            rbt[k] = str(k)
        del rbt[2]
        self.assertNotIn(2, rbt)
        self.assertEqual(len(rbt), 4)

        # Deleting non‑existent key must raise
        with self.assertRaises(KeyError):
            del rbt[99]

    # ------------------------------------------------------------------
    #  Iteration / order
    # ------------------------------------------------------------------
    def test_inorder_iteration(self):
        data = [(7, "seven"), (3, "three"), (9, "nine"), (1, "one")]
        rbt = RedBlackTree[int, str](data)

        # Keys should be emitted in ascending order
        self.assertEqual(list(rbt), [1, 3, 7, 9])

        # Items should be sorted by key as well
        self.assertEqual(
            rbt.items(), [(1, "one"), (3, "three"), (7, "seven"), (9, "nine")]
        )

    def test_min_max(self):
        rbt = RedBlackTree[int, str]()
        keys = [15, 2, 40, 7, 30]
        for k in keys:
            rbt[k] = str(k)

        self.assertEqual(rbt.min_key(), 2)
        self.assertEqual(rbt.max_key(), 40)

    # ------------------------------------------------------------------
    #  Successor / predecessor
    # ------------------------------------------------------------------
    def test_successor_predecessor(self):
        rbt = RedBlackTree[int, str]()
        for k in [10, 20, 30, 40, 50]:
            rbt[k] = str(k)

        self.assertEqual(rbt.successor(20), 30)
        self.assertEqual(rbt.predecessor(20), 10)

        # Edge cases – no successor / predecessor
        with self.assertRaises(KeyError):
            rbt.successor(50)
        with self.assertRaises(KeyError):
            rbt.predecessor(10)

        # Asking for a non‑existent key should raise as well
        with self.assertRaises(KeyError):
            rbt.successor(999)

    # ------------------------------------------------------------------
    #  Randomised stress test vs. Python dict
    # ------------------------------------------------------------------
    def test_random_operations_against_dict(self):
        random.seed(12345)
        rbt = RedBlackTree[int, int]()
        reference = {}  # normal dict for ground truth

        ops = 10_000
        key_range = range(0, 500)

        for _ in range(ops):
            op = random.choice(["insert", "delete"])
            k = random.choice(key_range)
            if op == "insert":
                v = random.randint(-1_000, 1_000)
                rbt[k] = v
                reference[k] = v
            else:  # delete
                if k in reference:
                    del rbt[k]
                    del reference[k]

            # After each mutation, validate red‑black invariants
            if len(rbt) > 0:
                rbt.validate()

        # Final check – the entire contents must match
        self.assertEqual(set(rbt.keys()), set(reference.keys()))
        for k in reference:
            self.assertEqual(rbt[k], reference[k])

        # Verify iteration order matches sorted order of keys
        self.assertEqual(list(rbt), sorted(reference.keys()))

    # ------------------------------------------------------------------
    #  Explicit validation of invariants on a crafted tree
    # ------------------------------------------------------------------
    def test_validate_simple_tree(self):
        rbt = RedBlackTree[int, str]()
        for k in [10, 5, 15, 2, 7, 12, 20]:
            rbt[k] = str(k)

        # Should not raise
        rbt.validate()

        # Manually corrupt the tree and ensure validate detects it
        # (example: make root red)
        rbt._root.color = RED
        with self.assertRaises(AssertionError):
            rbt.validate()

        # Restore proper colour
        rbt._root.color = BLACK

        # Violate black‑height: make a leaf black while its sibling is red‑black
        # (pick a node with both children)
        node = rbt._root.left  # this is the node with key 5
        node.left.color = RED  # make leaf red
        with self.assertRaises(AssertionError):
            rbt.validate()

    # ------------------------------------------------------------------
    #  Clear operation (not in the public API but useful for internal tests)
    # ------------------------------------------------------------------
    def test_clear_by_deleting_all(self):
        rbt = RedBlackTree[int, str]()
        for i in range(20):
            rbt[i] = str(i)

        # Delete everything one by one
        for i in range(20):
            del rbt[i]

        self.assertEqual(len(rbt), 0)
        self.assertFalse(bool(rbt))

        # Even after deleting everything the sentinel is still healthy
        rbt.validate()


if __name__ == "__main__":
    unittest.main(verbosity=2)
