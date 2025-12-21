#!/usr/bin/env python3
"""
EXPERIMENT 02: Semantic Vector Clustering
==========================================

Hypothesis: Our 16D vectors should cluster semantically similar words.
- Emotions should cluster together
- Body parts should cluster together
- Animals should cluster together

Test: Check if known categories form tight clusters in our space.
"""

import json
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("EXPERIMENT 02: Semantic Vector Clustering")
print("=" * 60)

# Load semantic vectors
print("\n[1] Loading semantic vectors...")
data_path = Path(__file__).parent.parent.parent.parent / "data/semantic_vectors"

with open(data_path / "noun_vectors_16d.json") as f:
    noun_data = json.load(f)

dimensions = noun_data['dimensions']
noun_vectors = noun_data['vectors']
print(f"  Loaded {len(noun_vectors)} noun vectors with {len(dimensions)} dimensions")

# Define test categories
CATEGORIES = {
    'emotions': ['joy', 'sadness', 'anger', 'fear', 'love', 'hate',
                 'happiness', 'grief', 'rage', 'terror', 'affection', 'disgust'],

    'body_parts': ['head', 'hand', 'eye', 'heart', 'arm', 'leg',
                   'face', 'foot', 'finger', 'nose', 'mouth', 'ear'],

    'animals': ['dog', 'cat', 'horse', 'bird', 'fish', 'lion',
                'wolf', 'bear', 'snake', 'eagle', 'rabbit', 'deer'],

    'time': ['day', 'night', 'morning', 'evening', 'hour', 'minute',
             'week', 'month', 'year', 'moment', 'second', 'century'],

    'nature': ['tree', 'flower', 'river', 'mountain', 'forest', 'ocean',
               'sun', 'moon', 'star', 'sky', 'rain', 'wind'],

    'people': ['man', 'woman', 'child', 'father', 'mother', 'son',
               'daughter', 'friend', 'stranger', 'king', 'queen', 'soldier'],
}

# Get vectors for categories
print("\n[2] Extracting category vectors...")

category_vectors = {}
category_words = {}

for cat_name, words in CATEGORIES.items():
    vectors = []
    found_words = []

    for word in words:
        if word in noun_vectors:
            vectors.append(noun_vectors[word])
            found_words.append(word)

    if vectors:
        category_vectors[cat_name] = np.array(vectors)
        category_words[cat_name] = found_words
        print(f"  {cat_name}: {len(found_words)}/{len(words)} words found")

# Compute within-category and between-category similarities
print("\n[3] Computing cluster quality...")

def compute_similarity_stats(vectors):
    """Compute mean pairwise cosine similarity."""
    if len(vectors) < 2:
        return 0.0, 0.0

    sim_matrix = cosine_similarity(vectors)
    # Get upper triangle (excluding diagonal)
    upper_tri = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]

    return np.mean(upper_tri), np.std(upper_tri)


print(f"\n  {'Category':<12} {'Within-Sim':<12} {'Std':<8} {'Interpretation'}")
print("  " + "-" * 50)

within_sims = []
for cat_name, vectors in category_vectors.items():
    mean_sim, std_sim = compute_similarity_stats(vectors)
    within_sims.append(mean_sim)

    if mean_sim > 0.5:
        interp = "TIGHT cluster"
    elif mean_sim > 0.2:
        interp = "Moderate cluster"
    else:
        interp = "Loose/scattered"

    print(f"  {cat_name:<12} {mean_sim:>10.3f}   {std_sim:>6.3f}   {interp}")

# Compute between-category similarity
print("\n[4] Computing between-category separation...")

between_sims = []
cat_names = list(category_vectors.keys())

print(f"\n  {'Pair':<25} {'Between-Sim':<12}")
print("  " + "-" * 40)

for i, cat1 in enumerate(cat_names):
    for cat2 in cat_names[i+1:]:
        # Compute similarity between all pairs of words from different categories
        v1 = category_vectors[cat1]
        v2 = category_vectors[cat2]

        cross_sim = cosine_similarity(v1, v2)
        mean_between = np.mean(cross_sim)
        between_sims.append(mean_between)

        print(f"  {cat1}-{cat2:<15} {mean_between:>10.3f}")

# Summary statistics
print("\n" + "=" * 60)
print("CLUSTER QUALITY METRICS")
print("=" * 60)

avg_within = np.mean(within_sims)
avg_between = np.mean(between_sims)
separation = avg_within - avg_between

print(f"""
Average within-category similarity:  {avg_within:.3f}
Average between-category similarity: {avg_between:.3f}
Separation (within - between):       {separation:.3f}
""")

if separation > 0.1:
    print("✓ VALIDATED: Categories form distinct clusters")
    print("  → Semantic vectors capture categorical structure")
    validation = "PASSED"
elif separation > 0:
    print("~ PARTIAL: Some clustering, weak separation")
    validation = "PARTIAL"
else:
    print("✗ FAILED: No meaningful clustering")
    validation = "FAILED"

# Visualize with t-SNE
print("\n[5] Generating t-SNE visualization...")

all_vectors = []
all_labels = []
all_words = []

for cat_name, vectors in category_vectors.items():
    for i, vec in enumerate(vectors):
        all_vectors.append(vec)
        all_labels.append(cat_name)
        all_words.append(category_words[cat_name][i])

all_vectors = np.array(all_vectors)

if len(all_vectors) > 10:
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_vectors)-1))
    coords = tsne.fit_transform(all_vectors)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))

    colors = plt.cm.tab10(np.linspace(0, 1, len(CATEGORIES)))
    color_map = {cat: colors[i] for i, cat in enumerate(CATEGORIES.keys())}

    for i, (x, y) in enumerate(coords):
        ax.scatter(x, y, c=[color_map[all_labels[i]]], s=100, alpha=0.7)
        ax.annotate(all_words[i], (x, y), fontsize=8, alpha=0.8)

    # Legend
    for cat_name, color in color_map.items():
        ax.scatter([], [], c=[color], label=cat_name, s=100)
    ax.legend(loc='best')

    ax.set_title(f'Semantic Clusters (t-SNE)\nSeparation: {separation:.3f}')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'semantic_clusters_tsne.png', dpi=150)
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'semantic_clusters_tsne.png'}")

# Find nearest neighbors for test words
print("\n[6] Nearest neighbor test...")

test_words = ['love', 'death', 'king', 'dog', 'mountain']

for test_word in test_words:
    if test_word not in noun_vectors:
        continue

    test_vec = np.array(noun_vectors[test_word]).reshape(1, -1)

    # Compute similarity to all words
    similarities = []
    for word, vec in noun_vectors.items():
        if word != test_word:
            sim = cosine_similarity(test_vec, np.array(vec).reshape(1, -1))[0, 0]
            similarities.append((word, sim))

    similarities.sort(key=lambda x: -x[1])
    top_5 = similarities[:5]

    print(f"\n  '{test_word}' nearest neighbors:")
    for word, sim in top_5:
        print(f"    {word:<15} (sim={sim:.3f})")

# Save results
results = {
    'avg_within_similarity': float(avg_within),
    'avg_between_similarity': float(avg_between),
    'separation': float(separation),
    'validation': validation,
    'category_stats': {
        cat: {
            'n_words': len(category_words[cat]),
            'within_sim': float(compute_similarity_stats(category_vectors[cat])[0])
        }
        for cat in category_vectors
    }
}

with open(OUTPUT_DIR / 'semantic_clusters_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {OUTPUT_DIR / 'semantic_clusters_results.json'}")
