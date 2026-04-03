"""
================================================================================
APRIORI ALGORITHM - Real-World Example: Supermarket Basket Analysis
================================================================================
REAL-TIME USE CASE:
    A supermarket wants to discover which products are frequently bought TOGETHER
    to optimize store layout, create bundle deals, and recommend products.
    E.g., "Customers who buy Bread often also buy Butter."

ALGORITHM:
    Apriori finds frequent itemsets by:
    1. Count single items that meet minimum support threshold
    2. Generate candidate pairs from frequent singles
    3. Count pairs that meet minimum support
    4. Generate triples from frequent pairs, and so on...
    Then generates ASSOCIATION RULES with confidence and lift metrics.

MODEL TYPE AFTER TRAINING:
    -> A set of ASSOCIATION RULES: {antecedent} -> {consequent}
    -> Each rule has: support, confidence, and lift scores.
    -> Support = how often items appear together
    -> Confidence = P(consequent | antecedent)
    -> Lift = how much more likely vs random chance (>1 = positive association)
    -> Saved as rules, NOT a predictive model - a pattern discovery tool.
================================================================================
"""
import pandas as pd
import numpy as np
from collections import defaultdict
from itertools import combinations
import json
import os

def get_support(transactions, itemset):
    """Count how many transactions contain all items in the itemset."""
    count = sum(1 for t in transactions if itemset.issubset(t))
    return count / len(transactions)

def apriori(transactions, min_support=0.1):
    """Find all frequent itemsets using the Apriori algorithm."""
    # STEP A: Find frequent 1-itemsets
    items = set(item for t in transactions for item in t)
    freq_itemsets = {}
    for item in items:
        s = get_support(transactions, frozenset([item]))
        if s >= min_support:
            freq_itemsets[frozenset([item])] = s

    # STEP B: Build larger itemsets from frequent smaller ones
    k = 2
    current_freq = [fs for fs in freq_itemsets if len(fs) == 1]
    while current_freq:
        all_items = set(item for fs in current_freq for item in fs)
        candidates = [frozenset(c) for c in combinations(all_items, k)]
        new_freq = []
        for c in candidates:
            s = get_support(transactions, c)
            if s >= min_support:
                freq_itemsets[c] = s
                new_freq.append(c)
        current_freq = new_freq
        k += 1
    return freq_itemsets

def generate_rules(freq_itemsets, transactions, min_confidence=0.5):
    """Generate association rules from frequent itemsets."""
    rules = []
    for itemset, support in freq_itemsets.items():
        if len(itemset) < 2:
            continue
        for i in range(1, len(itemset)):
            for antecedent in combinations(itemset, i):
                antecedent = frozenset(antecedent)
                consequent = itemset - antecedent
                ant_support = get_support(transactions, antecedent)
                if ant_support > 0:
                    confidence = support / ant_support
                    if confidence >= min_confidence:
                        con_support = get_support(transactions, consequent)
                        lift = confidence / con_support if con_support > 0 else 0
                        rules.append({
                            "antecedent": sorted(antecedent),
                            "consequent": sorted(consequent),
                            "support": round(support, 4),
                            "confidence": round(confidence, 4),
                            "lift": round(lift, 4),
                        })
    return sorted(rules, key=lambda x: x["lift"], reverse=True)

def main():
    # -------------------------------------------------------------------------
    # STEP 1: Load the demo dataset (supermarket transactions)
    # -------------------------------------------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(script_dir, "dataset.csv"))
    print("=== STEP 1: Dataset Loaded ===")
    print(f"Total transactions: {len(df)}")
    print(f"\nSample baskets:")
    for _, row in df.head(5).iterrows():
        print(f"  TX #{int(row['TransactionID'])}: {row['Items']}")

    # -------------------------------------------------------------------------
    # STEP 2: Parse transactions into sets of items
    # -------------------------------------------------------------------------
    transactions = [set(row["Items"].split(",")) for _, row in df.iterrows()]

    all_items = set(item for t in transactions for item in t)
    item_counts = defaultdict(int)
    for t in transactions:
        for item in t:
            item_counts[item] += 1

    print(f"\n=== STEP 2: Transaction Parsing ===")
    print(f"Unique items: {sorted(all_items)}")
    print(f"\nItem frequencies:")
    for item, count in sorted(item_counts.items(), key=lambda x: -x[1]):
        pct = count / len(transactions) * 100
        bar = "#" * int(pct / 3)
        print(f"  {item:10s}: {count:3d} ({pct:5.1f}%) {bar}")

    # -------------------------------------------------------------------------
    # STEP 3: Find frequent itemsets using Apriori
    # min_support=0.2 means items must appear in at least 20% of transactions
    # -------------------------------------------------------------------------
    min_support = 0.2
    freq_itemsets = apriori(transactions, min_support=min_support)

    print(f"\n=== STEP 3: Frequent Itemsets (min_support={min_support}) ===")
    print(f"Found: {len(freq_itemsets)} frequent itemsets")
    for itemset, support in sorted(freq_itemsets.items(), key=lambda x: -x[1]):
        items_str = ", ".join(sorted(itemset))
        print(f"  {{{items_str}:25s}} -> support={support:.3f} ({support*100:.1f}%)")

    # -------------------------------------------------------------------------
    # STEP 4: Generate association rules
    # min_confidence=0.5 means the rule must be right at least 50% of the time
    # -------------------------------------------------------------------------
    min_confidence = 0.5
    rules = generate_rules(freq_itemsets, transactions, min_confidence=min_confidence)

    print(f"\n=== STEP 4: Association Rules (min_confidence={min_confidence}) ===")
    print(f"Found: {len(rules)} rules")
    print(f"\n{'Rule':>40s} | {'Support':>8s} | {'Confidence':>10s} | {'Lift':>6s}")
    print("-" * 75)
    for rule in rules[:15]:
        ant = ", ".join(rule["antecedent"])
        con = ", ".join(rule["consequent"])
        rule_str = f"{{{ant}}} -> {{{con}}}"
        print(f"{rule_str:>40s} | {rule['support']:>8.3f} | {rule['confidence']:>10.3f} | {rule['lift']:>6.2f}")

    # -------------------------------------------------------------------------
    # STEP 5: Interpret the rules for business decisions
    # -------------------------------------------------------------------------
    print(f"\n=== STEP 5: Business Insights ===")
    for i, rule in enumerate(rules[:5], 1):
        ant = " + ".join(rule["antecedent"])
        con = " + ".join(rule["consequent"])
        print(f"\n  Rule {i}: '{ant}' -> '{con}'")
        print(f"    Support    : {rule['support']*100:.1f}% of all transactions contain both")
        print(f"    Confidence : {rule['confidence']*100:.1f}% of '{ant}' buyers also buy '{con}'")
        if rule["lift"] > 1:
            print(f"    Lift       : {rule['lift']:.2f}x more likely than random -> STRONG association")
        else:
            print(f"    Lift       : {rule['lift']:.2f}x -> weak/no association")

    # -------------------------------------------------------------------------
    # STEP 6: Save the rules
    # -------------------------------------------------------------------------
    rules_path = os.path.join(script_dir, "association_rules.json")
    with open(rules_path, "w") as f:
        json.dump(rules, f, indent=2)

    print(f"\n=== STEP 6: Rules Saved ===")
    print(f"Saved to: {rules_path}")
    print(f"\n--- WHAT IS THE TRAINED MODEL? ---")
    print(f"Type     : Apriori (Association Rule Mining)")
    print(f"Output   : {len(rules)} association rules (NOT a predictive model)")
    print(f"Contents : Rules like '{{Bread}} -> {{Butter}}' with support/confidence/lift")
    print(f"Purpose  : Discover item CO-OCCURRENCE patterns")
    print(f"NOT      : A classifier or regressor - no predictions on new data")
    print(f"Use for  : Product placement, bundle deals, cross-selling recommendations")

if __name__ == "__main__":
    main()
