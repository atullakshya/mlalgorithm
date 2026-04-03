"""
================================================================================
FP-GROWTH - Real-World Example: Electronics Store Cross-Selling
================================================================================
REAL-TIME USE CASE:
    An electronics store wants to find which accessories customers buy with
    main products (e.g., Laptop+Mouse+Keyboard) for cross-selling recommendations.

ALGORITHM:
    FP-Growth (Frequent Pattern Growth) is FASTER than Apriori because:
    1. Builds a compressed FP-TREE of all transactions (one scan of data)
    2. Mines patterns from the tree using conditional pattern bases
    3. Never generates candidate itemsets (unlike Apriori)
    Much more efficient on large datasets.

MODEL TYPE AFTER TRAINING:
    -> Same as Apriori: FREQUENT PATTERNS + ASSOCIATION RULES
    -> But found MUCH FASTER using FP-Tree data structure
    -> Not a predictive model - a pattern discovery tool
    -> Saved as JSON rules with support/confidence/lift
================================================================================
"""
import pandas as pd
import numpy as np
from collections import defaultdict
import json
import os

class FPNode:
    """A node in the FP-Tree."""
    def __init__(self, item=None, count=0, parent=None):
        self.item = item
        self.count = count
        self.parent = parent
        self.children = {}
        self.next = None  # link to next node with same item

    def increment(self, count=1):
        self.count += count

class FPTree:
    """Frequent Pattern Tree - compressed representation of transactions."""
    def __init__(self, transactions, min_support):
        self.min_support = min_support
        self.header_table = {}
        self.root = FPNode()

        # Count item frequencies
        item_counts = defaultdict(int)
        for trans in transactions:
            for item in trans:
                item_counts[item] += 1

        # Keep only frequent items
        self.freq_items = {item: count for item, count in item_counts.items()
                          if count >= min_support}
        if not self.freq_items:
            return

        # Insert filtered & sorted transactions into tree
        for trans in transactions:
            filtered = [item for item in trans if item in self.freq_items]
            filtered.sort(key=lambda x: self.freq_items[x], reverse=True)
            if filtered:
                self._insert(filtered, self.root)

    def _insert(self, items, node):
        if not items:
            return
        first = items[0]
        if first in node.children:
            node.children[first].increment()
        else:
            new_node = FPNode(first, 1, node)
            node.children[first] = new_node
            if first in self.header_table:
                current = self.header_table[first]
                while current.next:
                    current = current.next
                current.next = new_node
            else:
                self.header_table[first] = new_node
        self._insert(items[1:], node.children[first])

def mine_patterns(tree, prefix, min_support):
    """Recursively mine frequent patterns from an FP-Tree."""
    patterns = {}
    for item in tree.header_table:
        new_pattern = prefix + [item]
        support = 0
        node = tree.header_table[item]
        cond_base = []
        while node:
            support += node.count
            path = []
            parent = node.parent
            while parent.item is not None:
                path.append(parent.item)
                parent = parent.parent
            if path:
                for _ in range(node.count):
                    cond_base.append(path)
            node = node.next
        patterns[frozenset(new_pattern)] = support
        cond_tree = FPTree(cond_base, min_support)
        if cond_tree.header_table:
            sub = mine_patterns(cond_tree, new_pattern, min_support)
            patterns.update(sub)
    return patterns

def main():
    # -------------------------------------------------------------------------
    # STEP 1: Load the demo dataset (electronics store transactions)
    # -------------------------------------------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(script_dir, "dataset.csv"))
    print("=== STEP 1: Dataset Loaded ===")
    print(f"Total transactions: {len(df)}")
    print(f"\nSample transactions:")
    for _, row in df.head(5).iterrows():
        print(f"  TX #{int(row['TransactionID'])}: {row['Items']}")

    # -------------------------------------------------------------------------
    # STEP 2: Parse transactions
    # -------------------------------------------------------------------------
    transactions = [row["Items"].split(",") for _, row in df.iterrows()]
    item_counts = defaultdict(int)
    for t in transactions:
        for item in t:
            item_counts[item] += 1

    print(f"\n=== STEP 2: Item Frequencies ===")
    for item, count in sorted(item_counts.items(), key=lambda x: -x[1]):
        pct = count / len(transactions) * 100
        bar = "#" * int(pct / 2)
        print(f"  {item:12s}: {count:3d} ({pct:5.1f}%) {bar}")

    # -------------------------------------------------------------------------
    # STEP 3: Build FP-Tree (compressed representation of all transactions)
    # This is the key advantage over Apriori - no candidate generation
    # -------------------------------------------------------------------------
    min_support_count = 5
    tree = FPTree(transactions, min_support_count)

    print(f"\n=== STEP 3: FP-Tree Built ===")
    print(f"Min support count : {min_support_count}")
    print(f"Frequent items    : {len(tree.freq_items)}")
    print(f"Header table items: {list(tree.header_table.keys())}")

    # -------------------------------------------------------------------------
    # STEP 4: Mine frequent patterns from the FP-Tree
    # -------------------------------------------------------------------------
    patterns = mine_patterns(tree, [], min_support_count)

    print(f"\n=== STEP 4: Frequent Patterns Mined ===")
    print(f"Total patterns found: {len(patterns)}")
    sorted_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)
    print(f"\nAll Frequent Patterns:")
    for pattern, support in sorted_patterns:
        items = ", ".join(sorted(pattern))
        pct = support / len(transactions) * 100
        print(f"  {{{items:35s}}} -> support={support:3d} ({pct:5.1f}%)")

    # -------------------------------------------------------------------------
    # STEP 5: Generate rules from patterns
    # -------------------------------------------------------------------------
    from itertools import combinations
    rules = []
    for itemset, support in patterns.items():
        if len(itemset) < 2:
            continue
        for i in range(1, len(itemset)):
            for ant in combinations(itemset, i):
                ant = frozenset(ant)
                con = itemset - ant
                ant_support = patterns.get(ant, 0)
                con_support = patterns.get(con, 0)
                if ant_support > 0 and con_support > 0:
                    confidence = support / ant_support
                    lift = confidence / (con_support / len(transactions))
                    if confidence >= 0.5:
                        rules.append({
                            "if_buy": sorted(ant),
                            "then_buy": sorted(con),
                            "support": round(support / len(transactions), 3),
                            "confidence": round(confidence, 3),
                            "lift": round(lift, 3),
                        })
    rules.sort(key=lambda x: x["lift"], reverse=True)

    print(f"\n=== STEP 5: Association Rules ({len(rules)} rules) ===")
    for rule in rules[:10]:
        ant = " + ".join(rule["if_buy"])
        con = " + ".join(rule["then_buy"])
        print(f"  Buy [{ant}] -> Also buy [{con}]")
        print(f"    Confidence: {rule['confidence']*100:.0f}%, Lift: {rule['lift']:.2f}x")

    # -------------------------------------------------------------------------
    # STEP 6: Save rules
    # -------------------------------------------------------------------------
    rules_path = os.path.join(script_dir, "fp_growth_rules.json")
    with open(rules_path, "w") as f:
        json.dump(rules, f, indent=2)

    print(f"\n=== STEP 6: Rules Saved ===")
    print(f"\n--- WHAT IS THE TRAINED MODEL? ---")
    print(f"Type      : FP-Growth (Frequent Pattern Mining)")
    print(f"Structure : FP-Tree (compressed transaction database)")
    print(f"Output    : {len(patterns)} frequent patterns + {len(rules)} association rules")
    print(f"vs Apriori: FP-Growth is FASTER (no candidate generation)")
    print(f"            - Apriori: generate candidates -> scan DB -> repeat")
    print(f"            - FP-Growth: build tree ONCE -> mine from tree")
    print(f"Use case  : Cross-selling, product bundling, recommendation systems")

if __name__ == "__main__":
    main()
