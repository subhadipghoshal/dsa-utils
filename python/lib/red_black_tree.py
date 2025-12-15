#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
red_black_tree.py
-----------------

A self‑balancing binary search tree based on the **Red‑Black** algorithm.
It behaves like a mutable mapping (key → value) while guaranteeing O(log n)
operations for insert, delete and lookup.

Features
~~~~~~~~
* `tree[key] = value`   – insert / replace
* `value = tree[key]`    – lookup (KeyError if missing)
* `del tree[key]`        – delete (KeyError if missing)
* `key in tree`          – membership test
* `len(tree)`            – number of stored items
* iteration (`for key in tree:`) – keys in ascending order
* `tree.items()`, `tree.keys()`, `tree.values()`
* `tree.min_key()`, `tree.max_key()`
* `tree.successor(key)`, `tree.predecessor(key)` (raise KeyError if not found)
* `tree.validate()` – sanity‑check that the red‑black invariants hold (useful for debugging)

The implementation uses a **single shared sentinel node** (`self._nil`) to
represent all leafs, which eliminates `None` checks everywhere and makes the
code easier to follow.

Typical usage
~~~~~~~~~~~~~
>>> from red_black_tree import RedBlackTree
>>> rbt = RedBlackTree()
>>> rbt[5] = "five"
>>> rbt[2] = "two"
>>> rbt[8] = "eight"
>>> rbt.min_key()
2
>>> rbt.max_key()
8
>>> sorted(rbt.items())
[(2, 'two'), (5, 'five'), (8, 'eight')]
>>> del rbt[5]
>>> 5 in rbt
False
"""

from __future__ import annotations

from typing import (
    Any,
    Callable,
    Generator,
    Iterable,
    List,
    Optional,
    Tuple,
    TypeVar,
    Generic,
)

# ----------------------------------------------------------------------
#  Type variables (keys must be comparable, values may be anything)
# ----------------------------------------------------------------------
K = TypeVar("K")
V = TypeVar("V")

# ----------------------------------------------------------------------
#  Node colour constants – using simple booleans is fastest
# ----------------------------------------------------------------------
RED = True
BLACK = False


class _Node(Generic[K, V]):
    """Internal node object – not meant to be used directly by callers."""

    __slots__ = ("key", "value", "color", "left", "right", "parent")

    def __init__(
        self,
        key: Optional[K] = None,
        value: Optional[V] = None,
        color: bool = BLACK,
        left: Optional["_Node[K, V]"] = None,
        right: Optional["_Node[K, V]"] = None,
        parent: Optional["_Node[K, V]"] = None,
    ) -> None:
        self.key = key
        self.value = value
        self.color = color
        self.left = left
        self.right = right
        self.parent = parent

    def __repr__(self) -> str:
        col = "R" if self.color == RED else "B"
        return f"<{col} {self.key!r}:{self.value!r}>"


class RedBlackTree(Generic[K, V]):
    """
    A mutable mapping implemented with a red‑black binary search tree.

    The public API mimics the built‑in ``dict`` where appropriate
    (``__setitem__``, ``__getitem__``, ``__delitem__``, ``__contains__``,
    ``__len__``, ``__iter__``).  All operations respect the O(log n) guarantees
    of a red‑black tree.
    """

    __slots__ = ("_root", "_nil", "_size")

    # ------------------------------------------------------------------
    #   Construction / basic container protocol
    # ------------------------------------------------------------------
    def __init__(self, items: Optional[Iterable[Tuple[K, V]]] = None) -> None:
        """
        Create an empty tree or optionally initialise it from an iterable of
        ``(key, value)`` pairs.

        Parameters
        ----------
        items : iterable of (key, value)   optional
            If supplied, each pair is inserted using ``insert`` (i.e. the
            whole operation is O(n log n)).
        """
        # The sentinel leaf node – shared by every leaf in the tree.
        self._nil: _Node[K, V] = _Node()
        self._nil.color = BLACK
        self._nil.left = self._nil.right = self._nil.parent = self._nil

        self._root: _Node[K, V] = self._nil
        self._size: int = 0

        if items is not None:
            for key, value in items:
                self[key] = value

    # ------------------------------------------------------------------
    #   Helper index look‑up (internal)
    # ------------------------------------------------------------------
    def _search_node(self, key: K) -> _Node[K, V]:
        """Return the node that holds *key* or the sentinel `_nil` if not found."""
        cur = self._root
        while cur is not self._nil:
            if key == cur.key:
                return cur
            elif key < cur.key:
                cur = cur.left
            else:
                cur = cur.right
        return self._nil

    # ------------------------------------------------------------------
    #   Public mapping methods
    # ------------------------------------------------------------------
    def __contains__(self, key: object) -> bool:  # type: ignore[override]
        return self._search_node(key) is not self._nil  # noqa: E721

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, key: K) -> V:
        node = self._search_node(key)
        if node is self._nil:
            raise KeyError(key)
        return node.value  # type: ignore[no-any-return]

    def __setitem__(self, key: K, value: V) -> None:
        """Insert *key* with *value* or replace the existing value."""
        self._bst_insert(key, value)

    def __delitem__(self, key: K) -> None:
        node = self._search_node(key)
        if node is self._nil:
            raise KeyError(key)
        self._delete_node(node)

    def __iter__(self) -> Generator[K, None, None]:
        """Yield keys in ascending order (in‑order traversal)."""
        stack: List[_Node[K, V]] = []
        cur: _Node[K, V] = self._root
        while stack or cur is not self._nil:
            while cur is not self._nil:
                stack.append(cur)
                cur = cur.left
            cur = stack.pop()
            yield cur.key  # type: ignore[no-any-return]
            cur = cur.right

    # ------------------------------------------------------------------
    #   Convenience collection‑like view methods
    # ------------------------------------------------------------------
    def keys(self) -> List[K]:
        """Return a list of all keys in sorted order."""
        return list(self)

    def values(self) -> List[V]:
        """Return a list of all values in key order."""
        return [self[key] for key in self]

    def items(self) -> List[Tuple[K, V]]:
        """Return a list of ``(key, value)`` pairs in sorted order."""
        return [(key, self[key]) for key in self]

    # ------------------------------------------------------------------
    #   Minimum / maximum helpers
    # ------------------------------------------------------------------
    def _minimum_node(self, start: Optional[_Node[K, V]] = None) -> _Node[K, V]:
        """Return the node with the smallest key in the subtree rooted at *start*."""
        node = start if start is not None else self._root
        if node is self._nil:
            raise ValueError("Tree is empty")
        while node.left is not self._nil:
            node = node.left
        return node

    def _maximum_node(self, start: Optional[_Node[K, V]] = None) -> _Node[K, V]:
        """Return the node with the largest key in the subtree rooted at *start*."""
        node = start if start is not None else self._root
        if node is self._nil:
            raise ValueError("Tree is empty")
        while node.right is not self._nil:
            node = node.right
        return node

    def min_key(self) -> K:
        """Return the smallest key stored in the tree."""
        return self._minimum_node().key  # type: ignore[no-any-return]

    def max_key(self) -> K:
        """Return the largest key stored in the tree."""
        return self._maximum_node().key  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    #   Successor / predecessor
    # ------------------------------------------------------------------
    def successor(self, key: K) -> K:
        """Return the smallest key greater than *key*; raise KeyError if none."""
        node = self._search_node(key)
        if node is self._nil:
            raise KeyError(key)

        if node.right is not self._nil:
            return self._minimum_node(node.right).key  # type: ignore[no-any-return]

        # Walk up until we find a node that is a left child of its parent.
        y = node.parent
        while y is not self._nil and node is y.right:
            node = y
            y = y.parent
        if y is self._nil:
            raise KeyError(f"No successor for {key}")
        return y.key  # type: ignore[no-any-return]

    def predecessor(self, key: K) -> K:
        """Return the greatest key smaller than *key*; raise KeyError if none."""
        node = self._search_node(key)
        if node is self._nil:
            raise KeyError(key)

        if node.left is not self._nil:
            return self._maximum_node(node.left).key  # type: ignore[no-any-return]

        y = node.parent
        while y is not self._nil and node is y.left:
            node = y
            y = y.parent
        if y is self._nil:
            raise KeyError(f"No predecessor for {key}")
        return y.key  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    #   Core BST insertion (without red‑black fixup yet)
    # ------------------------------------------------------------------
    def _bst_insert(self, key: K, value: V) -> None:
        """Standard BST insertion (no recolouring) – returns the new node."""
        parent = self._nil
        cur = self._root

        while cur is not self._nil:
            parent = cur
            if key == cur.key:
                # Key already exists → replace value, no tree‑structure change.
                cur.value = value
                return
            elif key < cur.key:
                cur = cur.left
            else:
                cur = cur.right

        # At this point `cur` is the sentinel, `parent` is where we attach.
        new_node = _Node(
            key=key,
            value=value,
            color=RED,
            left=self._nil,
            right=self._nil,
            parent=parent,
        )

        if parent is self._nil:
            self._root = new_node
        elif key < parent.key:
            parent.left = new_node
        else:
            parent.right = new_node

        self._size += 1
        self._fix_insert(new_node)

    # ------------------------------------------------------------------
    #   Insert fix‑up (preserves red‑black properties)
    # ------------------------------------------------------------------
    def _fix_insert(self, z: _Node[K, V]) -> None:
        """Restore red‑black properties after inserting node `z` (which is RED)."""
        while z.parent.color == RED:
            if z.parent is z.parent.parent.left:
                y = z.parent.parent.right  # uncle
                if y.color == RED:
                    # Case 1 – recolour
                    z.parent.color = BLACK
                    y.color = BLACK
                    z.parent.parent.color = RED
                    z = z.parent.parent
                else:
                    if z is z.parent.right:
                        # Case 2 – left‑rotate at parent
                        z = z.parent
                        self._rotate_left(z)
                    # Case 3 – right‑rotate at grandparent
                    z.parent.color = BLACK
                    z.parent.parent.color = RED
                    self._rotate_right(z.parent.parent)
            else:  # Mirror of the above (parent is a right child)
                y = z.parent.parent.left  # uncle
                if y.color == RED:
                    # Case 1 (mirror)
                    z.parent.color = BLACK
                    y.color = BLACK
                    z.parent.parent.color = RED
                    z = z.parent.parent
                else:
                    if z is z.parent.left:
                        # Case 2 (mirror)
                        z = z.parent
                        self._rotate_right(z)
                    # Case 3 (mirror)
                    z.parent.color = BLACK
                    z.parent.parent.color = RED
                    self._rotate_left(z.parent.parent)
        self._root.color = BLACK

    # ------------------------------------------------------------------
    #   Left / right rotations – helper primitives
    # ------------------------------------------------------------------
    def _rotate_left(self, x: _Node[K, V]) -> None:
        """Left‑rotate the subtree rooted at `x`."""
        y = x.right
        if y is self._nil:
            raise RuntimeError("rotate_left called on a node with nil right child")
        # Turn y's left subtree into x's right subtree
        x.right = y.left
        if y.left is not self._nil:
            y.left.parent = x
        # Link x's parent to y
        y.parent = x.parent
        if x.parent is self._nil:
            self._root = y
        elif x is x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        # Put x on y's left
        y.left = x
        x.parent = y

    def _rotate_right(self, y: _Node[K, V]) -> None:
        """Right‑rotate the subtree rooted at `y`."""
        x = y.left
        if x is self._nil:
            raise RuntimeError("rotate_right called on a node with nil left child")
        # Turn x's right subtree into y's left subtree
        y.left = x.right
        if x.right is not self._nil:
            x.right.parent = y
        # Link y's parent to x
        x.parent = y.parent
        if y.parent is self._nil:
            self._root = x
        elif y is y.parent.right:
            y.parent.right = x
        else:
            y.parent.left = x
        # Put y on x's right
        x.right = y
        y.parent = x

    # ------------------------------------------------------------------
    #   Deletion – public entry point
    # ------------------------------------------------------------------
    def _transplant(self, u: _Node[K, V], v: _Node[K, V]) -> None:
        """Replace subtree rooted at `u` with the subtree rooted at `v`."""
        if u.parent is self._nil:
            self._root = v
        elif u is u.parent.left:
            u.parent.left = v
        else:
            u.parent.right = v
        v.parent = u.parent

    def _delete_node(self, z: _Node[K, V]) -> None:
        """Delete the node `z` from the tree and fix up any colour violations."""
        y = z  # node to be spliced out
        y_original_color = y.color
        if z.left is self._nil:
            x = z.right
            self._transplant(z, z.right)
        elif z.right is self._nil:
            x = z.left
            self._transplant(z, z.left)
        else:
            # z has two children: find its in‑order successor `y`
            y = self._minimum_node(z.right)
            y_original_color = y.color
            x = y.right
            if y.parent is z:
                # Successor is directly the right child of `z`
                x.parent = y
            else:
                self._transplant(y, y.right)
                y.right = z.right
                y.right.parent = y
            self._transplant(z, y)
            y.left = z.left
            y.left.parent = y
            y.color = z.color

        self._size -= 1

        if y_original_color == BLACK:
            self._fix_delete(x)

    # ------------------------------------------------------------------
    #   Delete fix‑up (preserves red‑black properties)
    # ------------------------------------------------------------------
    def _fix_delete(self, x: _Node[K, V]) -> None:
        """
        Restore red‑black properties after deleting a black node.
        `x` is the node that moved into `y`'s original position (could be `nil`).
        """
        while x is not self._root and x.color == BLACK:
            if x is x.parent.left:
                w = x.parent.right  # sibling
                if w.color == RED:
                    # Case 1 – sibling is red
                    w.color = BLACK
                    x.parent.color = RED
                    self._rotate_left(x.parent)
                    w = x.parent.right
                if w.left.color == BLACK and w.right.color == BLACK:
                    # Case 2 – both of sibling's children are black
                    w.color = RED
                    x = x.parent
                else:
                    if w.right.color == BLACK:
                        # Case 3 – sibling's right child is black, left child is red
                        w.left.color = BLACK
                        w.color = RED
                        self._rotate_right(w)
                        w = x.parent.right
                    # Case 4 – sibling's right child is red
                    w.color = x.parent.color
                    x.parent.color = BLACK
                    w.right.color = BLACK
                    self._rotate_left(x.parent)
                    x = self._root
            else:
                # Mirror of the above, with "left" and "right" swapped
                w = x.parent.left
                if w.color == RED:
                    w.color = BLACK
                    x.parent.color = RED
                    self._rotate_right(x.parent)
                    w = x.parent.left
                if w.right.color == BLACK and w.left.color == BLACK:
                    w.color = RED
                    x = x.parent
                else:
                    if w.left.color == BLACK:
                        w.right.color = BLACK
                        w.color = RED
                        self._rotate_left(w)
                        w = x.parent.left
                    w.color = x.parent.color
                    x.parent.color = BLACK
                    w.left.color = BLACK
                    self._rotate_right(x.parent)
                    x = self._root
        x.color = BLACK

    # ------------------------------------------------------------------
    #   Validation/checking utilities – useful for debugging
    # ------------------------------------------------------------------
    def validate(self) -> None:
        """
        Verify that the tree satisfies all red‑black invariants.
        Raises ``AssertionError`` with a descriptive message if something is broken.
        """

        def dfs(node: _Node[K, V]) -> Tuple[int, bool]:
            """
            Return a pair ``(black_height, is_valid_subtree)``.
            The method raises on the first violation found.
            """
            if node is self._nil:
                return 1, True  # leaves count as black height 1 (they are black)

            # Property 1: node color is either RED or BLACK – enforced by type.

            # Property 2: root is black
            if node is self._root:
                assert node.color == BLACK, "Root is not black"

            # Property 3: red nodes have black children
            if node.color == RED:
                assert node.left.color == BLACK, "Red node has red left child"
                assert node.right.color == BLACK, "Red node has red right child"

            # BST ordering check
            if node.left is not self._nil:
                assert (
                    node.left.key < node.key
                ), "BST property violated (left child larger)"
            if node.right is not self._nil:
                assert (
                    node.right.key > node.key
                ), "BST property violated (right child smaller)"

            left_black, left_ok = dfs(node.left)
            right_black, right_ok = dfs(node.right)

            # Property 4: all paths have the same black height
            assert left_black == right_black, "Black-height mismatch"

            # Return black height of this subtree (add 1 if this node is black)
            bh = left_black + (1 if node.color == BLACK else 0)
            return bh, left_ok and right_ok

        # Trigger the recursive check from the root
        if self._root is not self._nil:
            dfs(self._root)

    # ------------------------------------------------------------------
    #   Convenience string representation (for debugging)
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        items = ", ".join(f"{k!r}: {v!r}" for k, v in self.items())
        return f"RedBlackTree({{{items}}})"
