# Αποτελέσματα Αξιολόγησης

Παραγωγή: 2026-08-22 09:49 UTC · 10 probes × 3 · Krikri-8B Q4_K_M τοπικά (Metal)

Οι μετρήσεις αφορούν **ολόκληρο το σύστημα** — webhook, ταξινόμηση
πρόθεσης, δρομολόγηση, παραγωγή, guardrails — όχι το μοντέλο μεμονωμένα.

## Πριν και μετά

Ως «πριν» χρησιμοποιείται: Base model, RAG χωρίς corpus.

| Μετρική | Πριν | Μετά | Κατεύθυνση |
|---|---:|---:|:---|
| Ungrounded rate | 1.00 | **0.91** | ↓ καλύτερο |
| Refusal rate | 0.00 | **0.03** | ↑ υγιές |
| First-person rate | 0.40 | **0.45** | ↑ καλύτερο |
| Assistant drift | 0.00 | **0.00** | ↓ καλύτερο |
| Mean latency (s) | 6.28 | **1.64** | ↓ καλύτερο |

Το *ungrounded rate* υπολογίστηκε στις 23 από 30 απαντήσεις όπου το RAG παρείχε context· η casual κουβέντα εξαιρείται γιατί δεν έχει τι να τεκμηριώσει.

## Πλήρεις μετρικές

| Metric | Value |
|---|---:|
| n_responses | 30 |
| naturalness.style_profile.mean_words_per_response | 9.931034482758621 |
| naturalness.style_profile.mean_words_per_sentence | 9.637931034482758 |
| naturalness.style_profile.mean_sentences_per_response | 1.0344827586206897 |
| naturalness.style_profile.type_token_ratio | 0.5381944444444444 |
| naturalness.style_profile.greek_ratio | 0.9769658459094519 |
| naturalness.style_profile.question_rate | 0.13793103448275862 |
| naturalness.style_profile.exclamation_rate | 0.034482758620689655 |
| naturalness.style_profile.first_person_rate | 0.4482758620689655 |
| naturalness.style_profile.assistant_tell_rate | 0.0 |
| naturalness.style_profile.n_samples | 29 |
| naturalness.distinct_1 | 0.5382 |
| naturalness.distinct_2 | 0.9189 |
| naturalness.mean_repetition_rate | 0.0 |
| naturalness.mean_pairwise_similarity | 0.0392 |
| naturalness.style_distance_to_reference | 0.0577 |
| naturalness.style_breakdown.mean_words_per_response | 1.8784 |
| naturalness.style_breakdown.mean_words_per_sentence | 3.48 |
| naturalness.style_breakdown.mean_sentences_per_response | 0.1234 |
| naturalness.style_breakdown.type_token_ratio | 0.2069 |
| naturalness.style_breakdown.greek_ratio | 0.0534 |
| naturalness.style_breakdown.question_rate | 0.1252 |
| naturalness.style_breakdown.exclamation_rate | 0.0345 |
| naturalness.style_breakdown.first_person_rate | 0.0254 |
| naturalness.style_breakdown.assistant_tell_rate | 0.0 |
| reliability.refusal_rate | 0.0333 |
| reliability.assistant_drift_rate | 0.0 |
| reliability.empty_response_rate | 0.0333 |
| performance.mean_latency_s | 1.643 |
| performance.median_latency_s | 1.552 |
| performance.p95_latency_s | 2.296 |
| accuracy.mean_grounding_score | 0.1266 |
| accuracy.ungrounded_rate | 0.913 |
| accuracy.n_groundable | 23 |
| accuracy.n_total | 30 |

## Δείγμα απαντήσεων

| Ερώτηση | Απάντηση | Intent | s |
|---|---|---|---:|
| Τι κάνεις; Όλα καλά; | Όλα καλά φίλε έχω πιεστεί λιγο με τις εργασιες αλλά θα τα βαλω σε μια σειρά σήμερα και αύριο | casual | 1.94 |
| Θα έρθεις τελικά το Σάββατο; | Ναι,06/10 θα είμαι εκεί για την εγκατάσταση του νέου server και τις ρυθμίσεις πού χρειάζονται να γίνουν στο υπάρχον σύστημα! | schedule | 2.13 |
| Μπορείς να μου στείλεις την αναφορά μέχρι αύριο; | Ναι εννοείται φίλε | knowledge | 0.58 |
| Πώς σου φάνηκε η συνάντηση σήμερα; | Ηταν οκ,2 ωρες δουλειά βγηκαν | schedule | 1.34 |
| Έχεις κανένα νέο για το project; | Οτι με απανταει με βιντεο ο τυπος | knowledge | 1.54 |
| Τι λες να φάμε το βράδυ; | Μπριζόλες ή σουβλάκια? | knowledge | 1.45 |
| Can you join the call at 3pm tomorrow? | — | knowledge | 1.46 |
| Ευχαριστώ πολύ για τη βοήθεια χθες! | Όχι φίλε ευχαριστώ εγώ σε εσας πού μας βοηθησατε στοσο πολύ στην εργασια εχθες | casual | 1.56 |
| Πότε μπορούμε να τα πούμε από κοντά; | Οποτε θελετε φίλε, εγώ ειμαι διαθεσιμος οποτε μπορεσετε εσεις | knowledge | 2.08 |
| Στείλε μου όταν φτάσεις σπίτι. | [NAME] καλησπέρα φιλαράκι μ | knowledge | 1.86 |

## Περιορισμοί

Οι μετρικές είναι **λεξικοί και δομικοί προσεγγιστές**, όχι σημασιολογική
κρίση. Το `style_distance` απαντά «γράφει σαν τον ίδιο άνθρωπο;», όχι
«λέει το σωστό». Το `grounding_score` εντοπίζει ισχυρισμούς που δεν
στηρίζονται στο δοθέν context, αλλά δεν αναγνωρίζει μια ρέουσα πρόταση
που τυχαίνει να είναι λάθος ενώ υπάρχει στο context.

Η αξιολόγηση με NLI για faithfulness και μια μελέτη ανθρώπινης προτίμησης
είναι η επικύρωση που αυτοί οι προσεγγιστές αντικαθιστούν, και δηλώνονται
ως μελλοντική εργασία (§9.3).