# Αποτελέσματα Αξιολόγησης

Παραγωγή: 2026-08-22 09:46 UTC · 10 probes × 3 · Krikri-8B Q4_K_M τοπικά (Metal)

Οι μετρήσεις αφορούν **ολόκληρο το σύστημα** — webhook, ταξινόμηση
πρόθεσης, δρομολόγηση, παραγωγή, guardrails — όχι το μοντέλο μεμονωμένα.

## Πριν και μετά

Ως «πριν» χρησιμοποιείται: Base model, RAG χωρίς corpus.

| Μετρική | Πριν | Μετά | Κατεύθυνση |
|---|---:|---:|:---|
| Ungrounded rate | 1.00 | **0.79** | ↓ καλύτερο |
| Refusal rate | 0.00 | **0.03** | ↑ υγιές |
| First-person rate | 0.40 | **0.55** | ↑ καλύτερο |
| Assistant drift | 0.00 | **0.00** | ↓ καλύτερο |
| Mean latency (s) | 6.28 | **1.57** | ↓ καλύτερο |

Το *ungrounded rate* υπολογίστηκε στις 24 από 30 απαντήσεις όπου το RAG παρείχε context· η casual κουβέντα εξαιρείται γιατί δεν έχει τι να τεκμηριώσει.

## Πλήρεις μετρικές

| Metric | Value |
|---|---:|
| n_responses | 30 |
| naturalness.style_profile.mean_words_per_response | 8.482758620689655 |
| naturalness.style_profile.mean_words_per_sentence | 7.879310344827586 |
| naturalness.style_profile.mean_sentences_per_response | 1.0689655172413792 |
| naturalness.style_profile.type_token_ratio | 0.556910569105691 |
| naturalness.style_profile.greek_ratio | 0.9809342230695901 |
| naturalness.style_profile.question_rate | 0.10344827586206896 |
| naturalness.style_profile.exclamation_rate | 0.06896551724137931 |
| naturalness.style_profile.first_person_rate | 0.5517241379310345 |
| naturalness.style_profile.assistant_tell_rate | 0.0 |
| naturalness.style_profile.n_samples | 29 |
| naturalness.distinct_1 | 0.5569 |
| naturalness.distinct_2 | 0.8848 |
| naturalness.mean_repetition_rate | 0.0 |
| naturalness.mean_pairwise_similarity | 0.0449 |
| naturalness.style_distance_to_reference | 0.0573 |
| naturalness.style_breakdown.mean_words_per_response | 0.4301 |
| naturalness.style_breakdown.mean_words_per_sentence | 1.7214 |
| naturalness.style_breakdown.mean_sentences_per_response | 0.0889 |
| naturalness.style_breakdown.type_token_ratio | 0.1882 |
| naturalness.style_breakdown.greek_ratio | 0.0574 |
| naturalness.style_breakdown.question_rate | 0.1597 |
| naturalness.style_breakdown.exclamation_rate | 0.069 |
| naturalness.style_breakdown.first_person_rate | 0.078 |
| naturalness.style_breakdown.assistant_tell_rate | 0.0 |
| reliability.refusal_rate | 0.0333 |
| reliability.assistant_drift_rate | 0.0 |
| reliability.empty_response_rate | 0.0333 |
| performance.mean_latency_s | 1.575 |
| performance.median_latency_s | 1.451 |
| performance.p95_latency_s | 2.476 |
| accuracy.mean_grounding_score | 0.2494 |
| accuracy.ungrounded_rate | 0.7917 |
| accuracy.n_groundable | 24 |
| accuracy.n_total | 30 |

## Δείγμα απαντήσεων

| Ερώτηση | Απάντηση | Intent | s |
|---|---|---|---:|
| Τι κάνεις; Όλα καλά; | Ειμαι καλά φίλε όλα ενταξει απλά άστα να πάνε ζεστη σήμερα | casual | 1.43 |
| Θα έρθεις τελικά το Σάββατο; | Ναι θα ερθω,7μιση μεχρι και μιση ώρα ειμαι οκ να μεινω γιατί έχω κ αλλες δουλειες μετά | schedule | 2.48 |
| Μπορείς να μου στείλεις την αναφορά μέχρι αύριο; | Ναι [NAME] δεν έχω θέμα, είναι έτοιμη και τώρα σε στέλνω με mail | knowledge | 2.21 |
| Πώς σου φάνηκε η συνάντηση σήμερα; | Ηταν οκ,5/10 θα έλεγα | schedule | 1.21 |
| Έχεις κανένα νέο για το project; | Οτι με απανταει με βιντεο | knowledge | 1.41 |
| Τι λες να φάμε το βράδυ; | Μπριζόλα ή κοτόπουλο ψητό? | knowledge | 1.4 |
| Can you join the call at 3pm tomorrow? | Ναι φίλε εννοείται | knowledge | 1.47 |
| Ευχαριστώ πολύ για τη βοήθεια χθες! | — | casual | 0.56 |
| Πότε μπορούμε να τα πούμε από κοντά; | Οποτε θες φίλε εγώ ειμαι ετοιμος οποτε μπορείς εσύ | knowledge | 1.94 |
| Στείλε μου όταν φτάσεις σπίτι. | Ετοιμος ειμαι φίλε | knowledge | 1.53 |

## Περιορισμοί

Οι μετρικές είναι **λεξικοί και δομικοί προσεγγιστές**, όχι σημασιολογική
κρίση. Το `style_distance` απαντά «γράφει σαν τον ίδιο άνθρωπο;», όχι
«λέει το σωστό». Το `grounding_score` εντοπίζει ισχυρισμούς που δεν
στηρίζονται στο δοθέν context, αλλά δεν αναγνωρίζει μια ρέουσα πρόταση
που τυχαίνει να είναι λάθος ενώ υπάρχει στο context.

Η αξιολόγηση με NLI για faithfulness και μια μελέτη ανθρώπινης προτίμησης
είναι η επικύρωση που αυτοί οι προσεγγιστές αντικαθιστούν, και δηλώνονται
ως μελλοντική εργασία (§9.3).