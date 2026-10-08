# 14 — Association rules

**Prerequisites:** sets, conditional probability (04), and basic counting. **Objective:** calculate support, confidence and lift; reproduce frequent itemset mining; and distinguish a co-occurrence rule from a causal claim.

A transaction is a set of items occurring within a defined event, such as one visit or one shopping basket. Repeated items within a transaction count once. Changing the event from one visit to a person's entire year changes the question and the resulting associations. Here input is a list of sets of string item names, rather than a numerical feature matrix. Empty transactions remain part of the population denominator.

For disjoint itemsets `A` and `B`, the rule `A → B` has support `count(A∪B)/N`, confidence `count(A∪B)/count(A)`, and lift `confidence/(count(B)/N)`. The union means that **all** required items occur together; it does not mean the logical event “A or B.” Leverage is `support-P(A)P(B)`, an absolute rather than relative measure of excess association.

The lesson uses six baskets: tea/sugar/biscuits; tea/biscuits; tea/sugar; coffee/sugar; tea/biscuits; coffee/biscuits. Tea occurs four times, biscuits four times, and their pair three times. Thus tea → biscuits has support `3/6=.5`, confidence `3/4=.75`, and lift `.75/(4/6)=1.125`. Its leverage is `.5-(4/6)(4/6)=1/18`. A confidence of .75 looks less impressive after noticing how common biscuits already are.

```bash
python -m mlfirst --lesson 14
```

`transaction_count` is six. `frequent_itemsets` lists item arrays with absolute support counts, using minimum count two. There are four frequent singletons and two frequent pairs: tea/biscuits and tea/sugar. `tea_to_biscuits` exposes all five statistics above along with the count. `rules` lists every rule generated from frequent unions that meets minimum confidence .5. Ordering is deterministic, so repeated runs are directly comparable.

Apriori avoids enumerating every possible combination. Adding an item cannot increase support: every basket containing `{tea,sugar,biscuits}` must also contain `{sugar,biscuits}`. If that subset fails the threshold, the larger set must fail too. The algorithm counts frequent singletons, joins them into candidate pairs, prunes using frequent subsets, and repeats at increasing sizes. This pruning is exact for the support criterion.

Low thresholds can still produce huge candidate sets and huge valid outputs. The implementation is intentionally simple and repeatedly scans the transactions; it is suitable for learning and small examples. Both itemset mining and rule generation use the same absolute minimum count. `association_rules` returns ordinary dictionaries, making the output easy to inspect or save.

Frequency is not causality. A purchasing pattern may arise from popularity, promotion, season, or repeated visits by the same people. Rare coincidences can have spectacular lift, especially after searching many candidates. Validate selected patterns on later or otherwise independent transactions. Frequent findings are not automatically useful decisions, and a rule's arrow does not establish temporal order.

**Practice**

1. Calculate tea → sugar confidence and lift.
2. If three empty baskets are appended, what happens to tea → biscuits support, confidence and lift?
3. Why can a three-item set with an infrequent two-item subset be discarded without counting it?

<details><summary>Worked solutions</summary>

1. Tea and sugar co-occur twice. Confidence is `2/4=.5`; sugar prevalence is `3/6=.5`, so lift is 1.
2. Counts stay fixed but N becomes 9. Support becomes `3/9=1/3`; confidence stays `3/4`; lift becomes `.75/(4/9)=27/16`. This demonstrates why transaction definition matters.
3. The larger event is contained within the subset event. Its support cannot exceed the subset's support, so it cannot reach the same minimum count.

</details>

**Scope and source:** Apriori and exhaustive rule generation from frequent sets are supplied. FP-growth, Eclat, closed/maximal itemset compression, and uncertainty-adjusted rule screening are not. See [module](../mlfirst/classical.py), [original Apriori](../book_code/companion_classical.py), and [exhaustive-reference tests](../tests/test_classical.py).
