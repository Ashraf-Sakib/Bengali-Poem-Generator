"""
src/evaluation.py
------------------
Comprehensive evaluation module for the Bangla Poetry Generator laboratory project.

Demonstrates and verifies:
1. Phase 1 & 2: Dataset & Corpus Preprocessing Statistics
2. Phase 2: Vocabulary Distribution & Lexical Density
3. Phase 3: Word2Vec Semantic Word Similarities & Theme Clusters
4. Phase 4: Scratch Trigram Language Model Counts & MLE Probabilities
5. Phase 5: Laplace Smoothing Comparison (Seen vs Unseen Trigrams) & Perplexity
6. Phase 6 & 7: Qualitative & Quantitative Generation Evaluation across Themes
"""

import os
import sys
import json
import time
from typing import List, Dict, Tuple, Any, Optional

from src.utils import (
    setup_utf8_output, Colors, print_banner, print_section,
    compute_poem_statistics, save_poem_to_file
)
from src.preprocessing import BengaliPreprocessor
from src.vocabulary import Vocabulary
from src.word2vec_model import BengaliWord2Vec
from src.trigram_model import TrigramLanguageModel
from src.smoothing import LaplaceSmoothing
from src.generator import BanglaPoemGenerator


class ProjectEvaluator:
    """
    Automated NLP Evaluation Suite for all project phases.
    """

    def __init__(
        self,
        preprocessor: BengaliPreprocessor,
        vocab: Vocabulary,
        w2v: BengaliWord2Vec,
        lm: TrigramLanguageModel,
        smoother: LaplaceSmoothing,
        generator: BanglaPoemGenerator
    ):
        self.preprocessor = preprocessor
        self.vocab = vocab
        self.w2v = w2v
        self.lm = lm
        self.smoother = smoother
        self.generator = generator

    def run_phase1_dataset_analysis(self) -> Dict[str, Any]:
        """Runs and displays Phase 1 Dataset Inspection."""
        print_section("Phase 1: Dataset Analysis (SAHITTO.ods)")
        summary = self.preprocessor.load_and_inspect_dataset()

        print(f"Dataset File Path        : {summary['file_path']}")
        print(f"Total Raw Records        : {summary['total_records']:,}")
        print(f"Available Columns        : {', '.join(summary['columns'])}")
        print(f"Text / Poem Column       : {summary['text_column']}")
        print(f"Author / Poet Column     : {summary['poet_column']}")
        print(f"Category / Label Column  : {summary['category_column']}")
        print(f"Missing Values           : {summary['missing_values']}")
        print(f"Exact Duplicate Rows     : {summary['exact_duplicates']}")
        print(f"Duplicate Poem Texts     : {summary['duplicate_poems']}")
        print(f"Unique Writers/Poets     : {summary['unique_writers']}")
        print(f"Top 5 Poets by Poem Count:")
        for writer, count in list(summary['top_writers'].items())[:5]:
            print(f"  - {writer:<25}: {count} poems")
        print(f"Category Distribution ({summary['unique_labels']} labels):")
        for label, count in list(summary['label_distribution'].items())[:6]:
            print(f"  - {label:<20}: {count} poems")

        print(f"\n{Colors.GREEN}Recommended Text Column for Training:{Colors.ENDC} {summary['recommended_text_column']}")
        return summary

    def run_phase2_preprocessing_evaluation(self) -> Dict[str, Any]:
        """Evaluates Phase 2 Preprocessing & Corpus Statistics."""
        print_section("Phase 2: Preprocessing & Corpus Statistics")
        stats = self.preprocessor.get_corpus_statistics()

        print(f"Total Processed Documents : {stats['total_documents']:,}")
        print(f"Total Poetic Verse Lines  : {stats['total_lines']:,}")
        print(f"Total Word Tokens         : {stats['total_tokens']:,}")
        print(f"Unique Vocabulary Size    : {stats['vocabulary_size']:,}")
        print(f"Type-Token Ratio (TTR)    : {stats['type_token_ratio']}")
        print(f"Average Words per Line    : {stats['average_tokens_per_line']}")
        print(f"Stop-Word Tokens Ratio    : {stats['stopword_tokens_ratio'] * 100:.1f}%")
        print(f"Content-Word Tokens Ratio : {stats['content_tokens_ratio'] * 100:.1f}%")

        print(f"\nTop 10 Most Frequent Words:")
        for rank, (word, freq) in enumerate(stats['top_20_frequent_words'][:10], 1):
            is_stop = " (Stopword)" if self.preprocessor.is_stopword(word) else ""
            print(f"  {rank:>2}. {word:<15} : {freq:>5} occurrences{is_stop}")

        return stats

    def run_phase3_word2vec_evaluation(self, test_keywords: Optional[List[str]] = None) -> Dict[str, Any]:
        """Evaluates Word2Vec semantic word representations and similarity lookups."""
        print_section("Phase 3: Word2Vec Semantic Word Embeddings")
        if test_keywords is None:
            test_keywords = ["নদী", "বৃষ্টি", "প্রকৃতি", "ভালোবাসা", "মৃত্যু"]

        results = {}
        for kw in test_keywords:
            sim_words = self.w2v.get_similar_words(kw, topn=5)
            results[kw] = sim_words
            print(f"\n{Colors.BOLD}Query Keyword: '{kw}'{Colors.ENDC}")
            if sim_words:
                print("  Top Semantically Similar Words (Cosine Similarity):")
                for w, score in sim_words:
                    bar = "█" * int(score * 20)
                    print(f"    • {w:<15} | Score: {score:.4f} | {bar}")
            else:
                print("  No direct similar words found (OOV).")

        return results

    def run_phase4_trigram_evaluation(self) -> Dict[str, Any]:
        """Evaluates Scratch Trigram Language Model counts and MLE probabilities."""
        print_section("Phase 4: Scratch Trigram Language Model (MLE)")

        print(f"Total Unigram Tokens Counted : {self.lm.total_words:,}")
        print(f"Unique Unigrams              : {len(self.lm.unigram_counts):,}")
        print(f"Unique Bigrams Observed      : {len(self.lm.bigram_counts):,}")
        unique_trigrams = sum(len(c) for c in self.lm.trigram_counts.values())
        print(f"Unique Trigrams Observed     : {unique_trigrams:,}")

        # Demonstrate sample MLE probability calculations
        sample_histories = [("আমার", "হৃদয়"), ("এই", "রাত"), ("জল", "আর")]
        print("\nSample MLE Conditional Continuations:")
        history_results = {}
        for w1, w2 in sample_histories:
            c_hist = self.lm.history_2_counts.get((w1, w2), 0)
            continuations = self.lm.get_candidate_continuations(w1, w2)
            top_conts = sorted(continuations.items(), key=lambda x: x[1], reverse=True)[:4]
            print(f"\n  Context: ({w1}, {w2}) [History Count: {c_hist}]")
            history_results[f"({w1}, {w2})"] = []
            if top_conts:
                for w3, c_tri in top_conts:
                    p_mle = self.lm.get_mle_trigram_prob(w1, w2, w3)
                    print(f"    -> w3: '{w3:<10}' | C(w1,w2,w3) = {c_tri:>3} | P_MLE = {p_mle:.5f}")
                    history_results[f"({w1}, {w2})"].append({"word": w3, "count": c_tri, "p_mle": p_mle})
            else:
                print("    (No continuations observed in training data)")

        return history_results

    def run_phase5_smoothing_evaluation(self) -> List[Dict[str, Any]]:
        """Evaluates Laplace Smoothing vs Unsmoothed MLE on seen and unseen trigrams."""
        print_section("Phase 5: Laplace Smoothing & Probability Estimation")

        # Pick both seen and synthetic unseen trigrams
        test_trigrams = [
            ("আমার", "হৃদয়", "মাঝে"),     # Frequently seen
            ("আমার", "হৃদয়", "নদী"),      # Seen history, possible unseen continuation
            ("এই", "রাত", "শেষে"),       # Seen
            ("মেঘের", "ডানায়", "আলো"),    # Literary combination
            ("নদী", "রোবট", "কম্পিউটার"),  # Unseen modern/synthetic history
            ("বৃষ্টি", "ভেজা", "দুপুর")     # Seen
        ]

        comparison_table = self.smoother.compare_mle_vs_smoothed(test_trigrams)

        print(f"{'Trigram Sequence':<28} | {'C(tri)':<6} | {'C(hist)':<6} | {'P_MLE':<10} | {'P_Laplace':<12} | {'Status'}")
        print("-" * 88)
        for r in comparison_table:
            tri_str = r['trigram']
            print(f"{tri_str:<28} | {r['trigram_count']:<6} | {r['history_count']:<6} | {r['p_mle']:<10.6f} | {r['p_smoothed']:<12.6f} | {r['status']}")

        print(f"\n{Colors.YELLOW}Key Observation:{Colors.ENDC}")
        print("  Notice that while P_MLE assigns exactly 0.000000 to unseen trigrams,")
        print("  Laplace Smoothing successfully assigns a non-zero probability floor,")
        print("  preventing division by zero and infinite cross-entropy/perplexity.")

        return comparison_table

    def run_phase6_7_generation_samples(self, keywords: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Generates and evaluates poems across multiple themes."""
        print_section("Phase 6 & 7: Poem Generation Showcase across Themes")

        if keywords is None:
            keywords = ["নদী", "প্রকৃতি", "বৃষ্টি", "ভালোবাসা", "জীবন"]

        generated_samples = []
        for kw in keywords:
            poem_result = self.generator.generate_poem(
                keyword=kw,
                num_lines=4,
                num_candidates=20,
                min_words_per_line=5,
                max_words_per_line=8,
                top_k=5,
                temperature=0.7
            )
            stats = compute_poem_statistics(poem_result['lines'])
            poem_result['poetic_metrics'] = stats
            generated_samples.append(poem_result)
            scores = poem_result['scoring_details']

            rhyme_info = scores.get('rhyme_details', [])
            pairs_str = " | ".join([f"({p['words'][0]} ↔ {p['words'][1]} [{p['coda1']}])" for p in rhyme_info if p.get('is_rhyme')])

            print(f"\n{Colors.BOLD}{Colors.YELLOW}═══ Theme: {kw} (Best-of-20 & AABB Rhyme) ═══{Colors.ENDC}")
            print(poem_result['poem_text'])
            print(f"{Colors.CYAN}Score Breakdown:{Colors.ENDC} Total: {scores['total_score']} | Fluency: {scores['fluency_log_prob']} | Relevance: {scores.get('thematic_relevance', 0.0)} | RhymeScore: {scores.get('rhyme_score', 0.0)} [{pairs_str}] | RepPenalty: -{scores.get('repetition_penalty', 0.0)}")
            print(f"{Colors.CYAN}Metrics:{Colors.ENDC} Words: {stats['total_words']} | Words/Line: {stats['avg_words_per_line']} | TTR: {stats['lexical_richness_ttr']} | Avg Perplexity: {poem_result['metrics']['average_perplexity']}")

            # Save to disk
            save_poem_to_file(
                poem_result['poem_text'],
                keyword=kw,
                metadata={
                    "lines": 4,
                    "best_of": 20,
                    "total_score": scores['total_score'],
                    "fluency_log_prob": scores['fluency_log_prob'],
                    "thematic_relevance": scores.get('thematic_relevance', 0.0),
                    "rhyme_score": scores.get('rhyme_score', 0.0),
                    "avg_pp": poem_result['metrics']['average_perplexity'],
                    "ttr": stats['lexical_richness_ttr']
                }
            )

        return generated_samples

    def run_full_evaluation_suite(self) -> Dict[str, Any]:
        """Executes all phase evaluations sequentially."""
        print_banner("Bangla Poetry Generator: NLP Laboratory Evaluation Suite", "Trigram Language Model + Laplace Smoothing + Word2Vec")

        t0 = time.time()
        p1 = self.run_phase1_dataset_analysis()
        p2 = self.run_phase2_preprocessing_evaluation()
        p3 = self.run_phase3_word2vec_evaluation()
        p4 = self.run_phase4_trigram_evaluation()
        p5 = self.run_phase5_smoothing_evaluation()
        p6 = self.run_phase6_7_generation_samples()
        elapsed = round(time.time() - t0, 2)

        print(f"\n{Colors.GREEN}✔ Full Evaluation Suite Completed in {elapsed} seconds.{Colors.ENDC}")

        return {
            "elapsed_seconds": elapsed,
            "phase1_dataset": p1,
            "phase2_preprocessing": p2,
            "phase3_word2vec": p3,
            "phase4_trigram": p4,
            "phase5_smoothing": p5,
            "phase6_generation": p6
        }
