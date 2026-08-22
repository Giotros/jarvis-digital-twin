# Αποτελέσματα Αξιολόγησης

Παραγωγή: 2026-08-22 15:02 UTC · 10 probes × 3 · Krikri-8B Q4_K_M τοπικά (Metal)

Οι μετρήσεις αφορούν **ολόκληρο το σύστημα** — webhook, ταξινόμηση
πρόθεσης, δρομολόγηση, παραγωγή, guardrails — όχι το μοντέλο μεμονωμένα.

## Πριν και μετά

Ως «πριν» χρησιμοποιείται: Base model, RAG χωρίς corpus.

| Μετρική | Πριν | Μετά | Κατεύθυνση |
|---|---:|---:|:---|
| Ungrounded rate | 1.00 | **1.00** | ↓ καλύτερο |
| Refusal rate | 0.00 | **0.00** | ↑ υγιές |
| First-person rate | 0.40 | **0.71** | ↑ καλύτερο |
| Assistant drift | 0.00 | **0.00** | ↓ καλύτερο |
| Mean latency (s) | 6.28 | **1.16** | ↓ καλύτερο |

Το *ungrounded rate* υπολογίστηκε στις 4 από 30 απαντήσεις όπου το RAG παρείχε context· η casual κουβέντα εξαιρείται γιατί δεν έχει τι να τεκμηριώσει.

## Πλήρεις μετρικές

| Metric | Value |
|---|---:|
| n_responses | 30 |
| naturalness.style_profile.mean_words_per_response | 13.083333333333334 |
| naturalness.style_profile.mean_words_per_sentence | 9.13888888888889 |
| naturalness.style_profile.mean_sentences_per_response | 1.4166666666666667 |
| naturalness.style_profile.type_token_ratio | 0.5477707006369427 |
| naturalness.style_profile.greek_ratio | 0.9916839916839917 |
| naturalness.style_profile.question_rate | 0.20833333333333334 |
| naturalness.style_profile.exclamation_rate | 0.125 |
| naturalness.style_profile.first_person_rate | 0.7083333333333334 |
| naturalness.style_profile.assistant_tell_rate | 0.0 |
| naturalness.style_profile.n_samples | 24 |
| naturalness.distinct_1 | 0.5478 |
| naturalness.distinct_2 | 0.9345 |
| naturalness.mean_repetition_rate | 0.0 |
| naturalness.mean_pairwise_similarity | 0.0577 |
| naturalness.style_distance_to_reference | 0.1031 |
| naturalness.style_breakdown.mean_words_per_response | 5.0307 |
| naturalness.style_breakdown.mean_words_per_sentence | 2.981 |
| naturalness.style_breakdown.mean_sentences_per_response | 0.2588 |
| naturalness.style_breakdown.type_token_ratio | 0.1973 |
| naturalness.style_breakdown.greek_ratio | 0.0681 |
| naturalness.style_breakdown.question_rate | 0.0548 |
| naturalness.style_breakdown.exclamation_rate | 0.125 |
| naturalness.style_breakdown.first_person_rate | 0.2346 |
| naturalness.style_breakdown.assistant_tell_rate | 0.0 |
| reliability.refusal_rate | 0.0 |
| reliability.assistant_drift_rate | 0.0 |
| reliability.empty_response_rate | 0.2 |
| performance.mean_latency_s | 1.159 |
| performance.median_latency_s | 0.973 |
| performance.p95_latency_s | 2.624 |
| accuracy.mean_grounding_score | 0.0 |
| accuracy.ungrounded_rate | 1.0 |
| accuracy.n_groundable | 4 |
| accuracy.n_total | 30 |

## Δείγμα απαντήσεων

| Ερώτηση | Απάντηση | Intent | s |
|---|---|---|---:|
| Τι κάνεις; Όλα καλά; | — | error | 0.04 |
| Θα έρθεις τελικά το Σάββατο; | — | error | 0.04 |
| Μπορείς να μου στείλεις την αναφορά μέχρι αύριο; | — | error | 0.02 |
| Πώς σου φάνηκε η συνάντηση σήμερα; | — | error | 0.02 |
| Έχεις κανένα νέο για το project; | — | error | 0.04 |
| Τι λες να φάμε το βράδυ; | Φάγαμε πολύ σήμερα μάλλον θα πάμε για μπύρες καλύτερα | casual | 1.12 |
| Can you join the call at 3pm tomorrow? | Ναι εννοείται να συμμετάσχω στην κλήση αύριο στις 15:00. Δεν έχω κάποια δέσμευση εκείνη την ώρα. | casual | 1.71 |
| Ευχαριστώ πολύ για τη βοήθεια χθες! | Πραγματικά τίποτα συ | casual | 0.64 |
| Πότε μπορούμε να τα πούμε από κοντά; | Εγώ δουλεύω μέχρι τις 2 σήμερα και αύριο έχω εξεταστική το πρωί οπότε θα φύγω λίγο πιο νωρίς για σπίτι. Κατά τις 6 μπορείς αν θες να βρεθούμε στο καφέ απέναντι από τη σχολή? | casual | 2.62 |
| Στείλε μου όταν φτάσεις σπίτι. | — | casual | 0.6 |

## Περιορισμοί

Οι μετρικές είναι **λεξικοί και δομικοί προσεγγιστές**, όχι σημασιολογική
κρίση. Το `style_distance` απαντά «γράφει σαν τον ίδιο άνθρωπο;», όχι
«λέει το σωστό». Το `grounding_score` εντοπίζει ισχυρισμούς που δεν
στηρίζονται στο δοθέν context, αλλά δεν αναγνωρίζει μια ρέουσα πρόταση
που τυχαίνει να είναι λάθος ενώ υπάρχει στο context.

Η αξιολόγηση με NLI για faithfulness και μια μελέτη ανθρώπινης προτίμησης
είναι η επικύρωση που αυτοί οι προσεγγιστές αντικαθιστούν, και δηλώνονται
ως μελλοντική εργασία (§9.3).