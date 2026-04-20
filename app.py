from flask import Flask, render_template, request, jsonify
import json
import re
import math
from collections import defaultdict

app = Flask(__name__, template_folder='.', static_folder='.', static_url_path='/')

# ── In-memory store ───────────────────────────────────────────────────────────
findings_db = []
clusters_db = []

# ── Minimal TF-IDF vectorizer ─────────────────────────────────────────────────
# Minimal stop-word list — intentionally lean so domain terms survive.
# Removed generic MRM terms like "model", "risk", "finding" that were OVER-filtering.
STOP_WORDS = {
    'a','an','the','is','are','was','were','be','been','being','have','has',
    'had','do','does','did','will','would','could','should','may','might',
    'shall','must','can','need','dare','ought','used','to','of','in','for',
    'on','with','at','by','from','as','into','through','during','before',
    'after','above','below','between','out','off','over','under','again',
    'further','then','once','and','but','or','nor','so','yet','both',
    'either','neither','not','only','own','same','than','too','very','just',
    'because','if','while','although','though','since','unless','until',
    'that','this','these','those','it','its','such','no','each','every',
    'all','any','few','more','most','other','some','what','which',
    'who','whom','there','their','they','we','our','us','you','your','he',
    'she','him','her','his','i','me','my','also','however','therefore',
    'thus','hence','moreover','furthermore','additionally','based','upon',
    'lead','leads','result','results','increase','increases','lack','lacks',
    'absence','without','ensure','including','across','within','where',
    'when','how','well','due','per','its','has','been',
}

# ── Domain-aware field weights ────────────────────────────────────────────────
# model_theme is the single strongest signal — weight it heavily.
# Title is highly discriminative (concise, intent-dense).
# Description carries the most raw semantic content for accurate separation.
# Business justification adds supporting regulatory/risk context.
# suggested_remediation is optional and lightly weighted to avoid skewing
# clusters purely on action-language when the field may be absent.
FIELD_WEIGHTS = {
    'model_theme': 8,
    'title': 5,
    'description': 8,
    'business_justification': 6,
    'suggested_remediation': 1,
}

# ── Known MRM theme → canonical group mapping ─────────────────────────────────
# This lets us seed clustering with structural knowledge when a theme is present.
THEME_GROUPS = {
    'modelling input data':  'data_quality',
    'modeling input data':   'data_quality',
    'model input data':      'data_quality',
    'input data':            'data_quality',
    'model documentation':   'documentation',
    'documentation':         'documentation',
    'model governance':      'governance',
    'governance':            'governance',
    'model methodology':     'methodology',
    'methodology':           'methodology',
    'model performance':     'performance_monitoring',
    'performance':           'performance_monitoring',
    'model implementation':  'implementation',
    'implementation':        'implementation',
}

def normalize_theme(theme: str) -> str:
    if not theme:
        return ''
    return THEME_GROUPS.get(theme.lower().strip(), theme.lower().strip())


def tokenize(text: str) -> list:
    text = text.lower()
    tokens = re.findall(r'\b[a-z][a-z\-]{2,}\b', text)
    return [t for t in tokens if t not in STOP_WORDS]


def build_weighted_doc(finding: dict) -> str:
    """
    Build a composite document string that repeats each field
    in proportion to its weight, giving heavier fields more
    influence on the final TF-IDF vector.
    """
    parts = []
    for field, weight in FIELD_WEIGHTS.items():
        val = finding.get(field, '') or ''
        # Normalise model_theme to canonical group before repeating
        if field == 'model_theme':
            val = normalize_theme(val)
        parts.extend([val] * weight)
    return ' '.join(parts)


def build_tfidf(docs: list, sublinear_tf: bool = True, bm25_b: float = 0.4):
    """
    TF-IDF with sublinear TF scaling and BM25-style length normalisation.

    sublinear_tf: replaces raw TF with 1 + log(TF), which dampens the
        dominance of very frequent terms and improves cluster separation.
    bm25_b: length-normalisation factor (0 = off, 1 = full). A modest
        value (0.4) penalises very long documents so that short, precise
        field repetitions (title, model_theme) are not drowned out.
    """
    N = len(docs)
    tokenized = [tokenize(d) for d in docs]
    df = defaultdict(int)
    for toks in tokenized:
        for t in set(toks):
            df[t] += 1
    vocab = list(df.keys())
    word2idx = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)

    # Average document length for BM25 length normalisation
    avg_dl = sum(len(t) for t in tokenized) / max(N, 1)

    vectors = []
    for toks in tokenized:
        dl = len(toks) or 1
        tf_raw = defaultdict(int)
        for t in toks:
            tf_raw[t] += 1
        vec = [0.0] * V
        for t, cnt in tf_raw.items():
            if t not in word2idx:
                continue
            # Sublinear TF
            tf = (1 + math.log(cnt)) if (sublinear_tf and cnt > 0) else (cnt / dl)
            # BM25-style length normalisation applied on top of TF
            tf_norm = tf / (1 - bm25_b + bm25_b * (dl / avg_dl))
            idf = math.log((N + 1) / (df[t] + 1)) + 1
            vec[word2idx[t]] = tf_norm * idf
        norm = math.sqrt(sum(x * x for x in vec)) or 1
        vectors.append([x / norm for x in vec])
    return vectors, vocab, word2idx


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b))


def kmeans(vectors, k, max_iter=300, n_restarts=5, seed=42):
    """
    K-Means with k-means++ initialisation and multiple restarts.
    Returns the labelling with the best within-cluster inertia.
    """
    import random
    n = len(vectors)
    if n == 0:
        return [], []
    k = min(k, n)

    best_labels = None
    best_inertia = float('inf')
    best_centroids = None

    for restart in range(n_restarts):
        random.seed(seed + restart * 31)

        # k-means++ initialisation
        centroids = [vectors[random.randint(0, n - 1)][:]]
        for _ in range(k - 1):
            dists = []
            for v in vectors:
                d = min(1 - cosine(v, c) for c in centroids)
                dists.append(max(d, 0))
            total = sum(dists) or 1
            r = random.random() * total
            cum = 0
            chosen = vectors[-1][:]
            for i, d in enumerate(dists):
                cum += d
                if cum >= r:
                    chosen = vectors[i][:]
                    break
            centroids.append(chosen)

        labels = [0] * n
        for _ in range(max_iter):
            new_labels = []
            for v in vectors:
                sims = [cosine(v, c) for c in centroids]
                new_labels.append(sims.index(max(sims)))
            if new_labels == labels:
                break
            labels = new_labels
            # Recompute centroids; reinitialise empty clusters
            for j in range(k):
                members = [vectors[i] for i, l in enumerate(labels) if l == j]
                if members:
                    centroids[j] = [sum(col) / len(members) for col in zip(*members)]
                else:
                    # Empty cluster: steal the point farthest from its centroid
                    farthest = max(range(n), key=lambda i: 1 - cosine(vectors[i], centroids[labels[i]]))
                    centroids[j] = vectors[farthest][:]

        # Compute inertia (sum of distances to assigned centroid)
        inertia = sum(1 - cosine(vectors[i], centroids[labels[i]]) for i in range(n))
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels[:]
            best_centroids = [c[:] for c in centroids]

    return best_labels, best_centroids


def bisecting_kmeans(vectors, k, seed=42):
    """
    Bisecting K-Means: repeatedly split the largest cluster using 2-means.

    This avoids the random initialisation sensitivity of standard K-Means for
    larger k values and tends to produce more balanced, semantically coherent
    clusters.  It is especially effective when the true clusters have unequal
    sizes, which is common in MRM finding datasets.
    """
    import random
    n = len(vectors)
    if n == 0:
        return [], []
    k = min(k, n)

    # Start with everything in one cluster
    clusters = [list(range(n))]

    while len(clusters) < k:
        # Pick the largest cluster to bisect (break ties by highest inertia)
        def cluster_inertia(idx_list):
            if len(idx_list) <= 1:
                return 0.0
            vecs = [vectors[i] for i in idx_list]
            centroid = [sum(col) / len(vecs) for col in zip(*vecs)]
            norm = math.sqrt(sum(x * x for x in centroid)) or 1
            centroid = [x / norm for x in centroid]
            return sum(1 - cosine(vectors[i], centroid) for i in idx_list)

        target_idx = max(range(len(clusters)),
                         key=lambda j: (len(clusters[j]), cluster_inertia(clusters[j])))
        target = clusters[target_idx]

        if len(target) < 2:
            break  # Cannot bisect a singleton

        sub_vecs = [vectors[i] for i in target]
        # Run 2-means with multiple restarts on the sub-cluster
        best_labels_2, _ = kmeans(sub_vecs, 2, max_iter=100, n_restarts=3,
                                  seed=seed + len(clusters) * 17)
        if not best_labels_2:
            break

        group_a = [target[i] for i, l in enumerate(best_labels_2) if l == 0]
        group_b = [target[i] for i, l in enumerate(best_labels_2) if l == 1]

        # Reject degenerate splits (one side empty)
        if not group_a or not group_b:
            break

        clusters.pop(target_idx)
        clusters.append(group_a)
        clusters.append(group_b)

    # Convert cluster list to a flat label array
    labels = [0] * n
    for cluster_label, idx_list in enumerate(clusters):
        for i in idx_list:
            labels[i] = cluster_label

    return labels, clusters


def _silhouette_score(vectors, labels, k):
    """
    Compute the mean silhouette coefficient for a clustering solution.
    Returns a value in [-1, 1]; higher is better.
    """
    n = len(vectors)
    if k <= 1 or n <= k:
        return -1.0

    # Group vectors by cluster
    clusters = defaultdict(list)
    for i, l in enumerate(labels):
        clusters[l].append(i)

    scores = []
    for i, label in enumerate(labels):
        same = [j for j in clusters[label] if j != i]
        if not same:
            scores.append(0.0)
            continue
        # Mean intra-cluster distance
        a = sum(1 - cosine(vectors[i], vectors[j]) for j in same) / len(same)
        # Mean nearest-cluster distance
        b_vals = []
        for other_label, members in clusters.items():
            if other_label == label:
                continue
            mean_dist = sum(1 - cosine(vectors[i], vectors[j]) for j in members) / len(members)
            b_vals.append(mean_dist)
        b = min(b_vals) if b_vals else 0.0
        denom = max(a, b)
        scores.append((b - a) / denom if denom > 0 else 0.0)

    return sum(scores) / len(scores) if scores else -1.0


def _gap_statistic(vectors, labels, k, n_refs=5, seed=99):
    """
    Compute a simplified gap statistic for the given clustering.

    Gap(k) = E[log W_k_ref] - log W_k
    where W_k is the within-cluster dispersion (sum of pairwise distances
    within clusters / 2*cluster_size) and W_k_ref is the same for random
    uniform reference datasets.

    A higher gap indicates that the clustering is meaningfully better than
    a random partition, signalling a good k.
    """
    import random

    def wcss(vecs, lbls, num_k):
        """Within-cluster sum of squared cosine distances."""
        total = 0.0
        for j in range(num_k):
            members = [vecs[i] for i, l in enumerate(lbls) if l == j]
            if len(members) < 2:
                continue
            # Sum of pairwise distances / (2 * |C|)
            s = sum(1 - cosine(members[a], members[b])
                    for a in range(len(members))
                    for b in range(a + 1, len(members)))
            total += s / (2 * len(members))
        return total or 1e-9

    W = wcss(vectors, labels, k)

    # Generate reference distributions: uniform samples in the feature hypercube
    dim = len(vectors[0]) if vectors else 1
    rng = random.Random(seed)
    ref_logs = []
    for _ in range(n_refs):
        ref_vecs = [[rng.uniform(-1, 1) for _ in range(dim)] for _ in range(len(vectors))]
        # Normalise
        ref_vecs = [[x / (math.sqrt(sum(v ** 2 for v in rv)) or 1) for x in rv] for rv in ref_vecs]
        ref_labels, _ = kmeans(ref_vecs, k, max_iter=50, n_restarts=1, seed=seed + _)
        if ref_labels:
            ref_logs.append(math.log(wcss(ref_vecs, ref_labels, k)))
    if not ref_logs:
        return 0.0
    return sum(ref_logs) / len(ref_logs) - math.log(W)


def _auto_select_k(vectors, n: int, has_themes: bool, theme_count: int) -> int:
    """
    Determine the optimal number of clusters using a combined
    silhouette + gap-statistic approach.

    Strategy:
    1. Run silhouette scoring across the candidate k range using bisecting
       K-Means (more stable than standard K-Means for auto-k selection).
    2. Compute the gap statistic for each candidate k to confirm that the
       chosen partition is meaningfully better than random.
    3. Pick the k that maximises a weighted composite of both scores,
       with a mild preference for compactness (smaller k) to avoid
       over-fragmentation on real-world MRM finding sets.

    Falls back gracefully for very small datasets.
    """
    if n <= 2:
        return 1
    if n == 3:
        return 2

    # Define search range based on dataset size
    if n <= 10:
        k_min, k_max = 2, max(2, n - 1)
    elif n <= 50:
        k_min, k_max = 2, min(12, n // 2)
    elif n <= 200:
        k_min, k_max = 2, min(25, n // 5)
    else:
        k_min, k_max = 2, min(50, n // 10)

    # If themes exist, include theme_count as a strong candidate
    if has_themes and theme_count >= k_min:
        k_max = max(k_max, min(theme_count + 2, n - 1))

    best_k = k_min
    best_composite = -float('inf')

    sil_scores = {}
    gap_scores = {}

    for k in range(k_min, k_max + 1):
        # Use bisecting K-Means for the primary k selection pass —
        # it is deterministic and avoids bad local minima.
        labels, _ = bisecting_kmeans(vectors, k)
        if not labels:
            continue

        sil = _silhouette_score(vectors, labels, k)
        sil_scores[k] = sil

        # Only compute gap for the plausible top candidates to keep it fast.
        # Use n_refs=3 for speed; accuracy is sufficient for our range sizes.
        gap = _gap_statistic(vectors, labels, k, n_refs=3)
        gap_scores[k] = gap

    if not sil_scores:
        return k_min

    # Normalise both scores to [0, 1] for the composite
    sil_min, sil_max = min(sil_scores.values()), max(sil_scores.values())
    gap_min, gap_max = min(gap_scores.values()), max(gap_scores.values())

    def norm(val, lo, hi):
        return (val - lo) / (hi - lo) if hi > lo else 0.5

    for k in range(k_min, k_max + 1):
        if k not in sil_scores:
            continue
        s_norm = norm(sil_scores[k], sil_min, sil_max)
        g_norm = norm(gap_scores[k], gap_min, gap_max)
        # Weight silhouette slightly higher; add a tiny compactness penalty
        composite = 0.6 * s_norm + 0.4 * g_norm - 0.005 * k
        if composite > best_composite:
            best_composite = composite
            best_k = k

    return best_k


def _estimate_k(vectors, n: int, has_themes: bool) -> int:
    """
    Determine optimal k automatically using silhouette scoring.
    - Counts distinct canonical theme groups as a hint for the search range.
    - Runs silhouette analysis across the plausible k range.
    - Returns the k that maximises cluster cohesion/separation.
    """
    theme_count = 0
    if has_themes:
        themes = set()
        for f in findings_db:
            t = normalize_theme(f.get('model_theme', ''))
            if t:
                themes.add(t)
        theme_count = len(themes)

    return _auto_select_k(vectors, n, has_themes, theme_count)


def _top_signals(fids):
    texts = []
    for fid in fids:
        f = next((x for x in findings_db if x['finding_id'] == fid), None)
        if f:
            texts.append(build_weighted_doc(f))
    tokens = []
    for t in texts:
        tokens.extend(tokenize(t))
    freq = defaultdict(int)
    for t in tokens:
        freq[t] += 1
    top = sorted(freq, key=lambda x: -freq[x])[:10]
    return ', '.join(top)


def _generate_why(fids):
    themes = []
    for fid in fids:
        f = next((x for x in findings_db if x['finding_id'] == fid), None)
        if f and f.get('model_theme'):
            themes.append(f['model_theme'])
    theme_str = ''
    if themes:
        unique = list(dict.fromkeys(themes))
        theme_str = f" under the theme(s): {', '.join(unique)}."
    return (
        f"Findings {', '.join(fids)} share overlapping concepts in their title, "
        f"description, and business justifications{theme_str} "
        "Bisecting K-Means clustering (with sublinear TF-IDF and BM25 length normalisation) "
        "detected strong semantic similarity in vocabulary, risk focus, and field content across all relevant fields."
    )


def run_clustering(k: int = None):
    global clusters_db
    if not findings_db:
        clusters_db = []
        return

    # Determine whether model_theme is available
    has_themes = any(f.get('model_theme') for f in findings_db)

    # Build weighted composite documents using all fields
    docs = [build_weighted_doc(f) for f in findings_db]

    vectors, vocab, word2idx = build_tfidf(docs)
    n = len(findings_db)

    if k is None:
        k = _estimate_k(vectors, n, has_themes)
    k = min(k, n)

    # Primary algorithm: bisecting K-Means — more stable and better at
    # separating theme-distinct clusters than a single K-Means run.
    labels, _ = bisecting_kmeans(vectors, k)

    # Fallback: if bisecting K-Means returns no labels, use standard K-Means
    if not labels:
        labels, _ = kmeans(vectors, k)

    cluster_map = defaultdict(list)
    for i, label in enumerate(labels):
        cluster_map[label].append(findings_db[i]['finding_id'])

    # ── Optional: sort clusters by dominant model_theme for stable labelling ──
    def cluster_theme_sort_key(fids):
        theme_counts = defaultdict(int)
        for fid in fids:
            f = next((x for x in findings_db if x['finding_id'] == fid), None)
            if f:
                theme_counts[normalize_theme(f.get('model_theme', ''))] += 1
        dominant = max(theme_counts, key=theme_counts.get) if theme_counts else ''
        theme_order = ['data_quality', 'documentation', 'governance', 'methodology', 'performance_monitoring']
        try:
            return theme_order.index(dominant)
        except ValueError:
            return 99

    sorted_clusters = sorted(cluster_map.values(), key=cluster_theme_sort_key)

    clusters_db = []
    for idx, fids in enumerate(sorted_clusters):
        clusters_db.append({
            'cluster_id': f"C{idx + 1}",
            'findings_included': fids,
            'why_grouped': _generate_why(fids),
            'semantic_signals': _top_signals(fids),
            'size': len(fids),
        })


def search_query(query: str) -> dict:
    if not clusters_db or not findings_db:
        return {'cluster': None, 'findings': [], 'score': 0}
    docs = [build_weighted_doc(f) for f in findings_db]
    all_docs = docs + [query]
    vectors, vocab, word2idx = build_tfidf(all_docs)
    q_vec = vectors[-1]
    doc_vecs = vectors[:-1]
    sims = [(cosine(q_vec, dv), i) for i, dv in enumerate(doc_vecs)]
    sims.sort(reverse=True)
    top_findings = [findings_db[i] for _, i in sims[:5]]
    top_ids = {f['finding_id'] for f in top_findings}
    best_cluster = None
    best_count = 0
    for c in clusters_db:
        overlap = len(set(c['findings_included']) & top_ids)
        if overlap > best_count:
            best_count = overlap
            best_cluster = c
    return {
        'cluster': best_cluster,
        'findings': top_findings,
        'score': round(sims[0][0], 3) if sims else 0,
    }


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/findings', methods=['GET'])
def get_findings():
    return jsonify(findings_db)


@app.route('/api/findings', methods=['POST'])
def add_finding():
    data = request.json
    required = ['finding_id', 'title', 'description', 'business_justification']
    if not all(k in data for k in required):
        return jsonify({'error': 'Missing required fields: finding_id, title, description, business_justification'}), 400
    if any(f['finding_id'] == data['finding_id'] for f in findings_db):
        return jsonify({'error': 'Finding ID already exists'}), 409
    record = {k: str(data[k]).strip() for k in required}
    if 'model_theme' in data and data['model_theme']:
        record['model_theme'] = str(data['model_theme']).strip()
    else:
        record['model_theme'] = ''
    if 'suggested_remediation' in data and data['suggested_remediation']:
        record['suggested_remediation'] = str(data['suggested_remediation']).strip()
    else:
        record['suggested_remediation'] = ''
    findings_db.append(record)
    run_clustering()
    return jsonify({'status': 'ok', 'total': len(findings_db)}), 201


@app.route('/api/findings/bulk', methods=['POST'])
def bulk_findings():
    rows = request.json
    added, skipped = 0, 0
    for row in rows:
        required = ['finding_id', 'title', 'description', 'business_justification']
        if not all(k in row for k in required):
            skipped += 1
            continue
        if any(f['finding_id'] == row['finding_id'] for f in findings_db):
            skipped += 1
            continue
        record = {k: str(row[k]).strip() for k in required}
        record['model_theme'] = str(row['model_theme']).strip() if row.get('model_theme') else ''
        record['suggested_remediation'] = str(row.get('suggested_remediation', '')).strip() if row.get('suggested_remediation') else ''
        findings_db.append(record)
        added += 1
    run_clustering()
    return jsonify({'added': added, 'skipped': skipped, 'total': len(findings_db)})


@app.route('/api/findings/upload', methods=['POST'])
def upload_findings():
    """Accept an Excel (.xlsx) or CSV file and bulk-import findings."""
    import io
    import pandas as pd

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    filename = file.filename.lower()

    try:
        if filename.endswith('.xlsx') or filename.endswith('.xls'):
            df = pd.read_excel(io.BytesIO(file.read()))
        elif filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(file.read()))
        else:
            return jsonify({'error': 'Unsupported file type. Please upload .xlsx or .csv'}), 400
    except Exception as e:
        return jsonify({'error': f'Failed to parse file: {str(e)}'}), 400

    # Normalise column names: strip whitespace, lowercase for matching
    df.columns = [c.strip() for c in df.columns]

    # Build a flexible column mapping (handle variations in casing/spacing)
    col_map = {}
    for col in df.columns:
        key = col.lower().replace(' ', '_').replace('-', '_')
        col_map[key] = col

    def resolve(candidates):
        for c in candidates:
            if c in col_map:
                return col_map[c]
        return None

    finding_id_col        = resolve(['finding_id', 'findingid', 'finding_no', 'id'])
    model_theme_col       = resolve(['model_theme', 'modeltheme', 'theme'])
    title_col             = resolve(['title'])
    description_col       = resolve(['description', 'desc'])
    business_just_col     = resolve(['business_justification', 'businessjustification',
                                     'business_justification', 'bj', 'justification'])
    remediation_col       = resolve(['suggested_remediation', 'remediation', 'suggestedremediation'])

    missing = [name for name, col in [
        ('Finding ID', finding_id_col),
        ('Title', title_col),
        ('Description', description_col),
        ('Business Justification', business_just_col),
    ] if col is None]

    if missing:
        return jsonify({
            'error': f'Required column(s) not found: {", ".join(missing)}. '
                     f'Expected headers: Finding ID, Model Theme, Title, Description, Business Justification. '
                     f'Found: {", ".join(df.columns.tolist())}'
        }), 400

    added, skipped = 0, 0
    for _, row in df.iterrows():
        fid   = str(row[finding_id_col]).strip()
        theme = str(row[model_theme_col]).strip() if model_theme_col and pd.notna(row.get(model_theme_col)) else ''
        title = str(row[title_col]).strip()
        desc  = str(row[description_col]).strip()
        bj    = str(row[business_just_col]).strip()
        rem   = str(row[remediation_col]).strip() if remediation_col and pd.notna(row.get(remediation_col)) else ''

        if not fid or not title or not desc or not bj or fid.lower() == 'nan':
            skipped += 1
            continue
        if any(f['finding_id'] == fid for f in findings_db):
            skipped += 1
            continue

        findings_db.append({
            'finding_id': fid,
            'model_theme': theme,
            'title': title,
            'description': desc,
            'business_justification': bj,
            'suggested_remediation': rem,
        })
        added += 1

    if added:
        run_clustering()

    return jsonify({'added': added, 'skipped': skipped, 'total': len(findings_db)})


@app.route('/api/findings/<fid>', methods=['DELETE'])
def delete_finding(fid):
    global findings_db
    findings_db = [f for f in findings_db if f['finding_id'] != fid]
    run_clustering()
    return jsonify({'status': 'ok'})


@app.route('/api/clusters', methods=['GET'])
def get_clusters():
    return jsonify(clusters_db)


@app.route('/api/clusters/rerun', methods=['POST'])
def rerun_clustering():
    k = request.json.get('k') if request.json else None
    run_clustering(k)
    return jsonify({'status': 'ok', 'clusters': len(clusters_db)})


@app.route('/api/search', methods=['POST'])
def search():
    query = (request.json or {}).get('query', '')
    if not query:
        return jsonify({'error': 'Empty query'}), 400
    result = search_query(query)
    return jsonify(result)


@app.route('/api/stats', methods=['GET'])
def stats():
    return jsonify({
        'total_findings': len(findings_db),
        'total_clusters': len(clusters_db),
    })


# ── Seed with sample data on first run ───────────────────────────────────────
SAMPLE_FINDINGS = [
    {"finding_id": "F001", "model_theme": "Modelling Input Data",
     "title": "Inadequate Data Quality Controls for Model Input Data",
     "description": "The model relies on multiple upstream source systems (including loan origination, customer master, and external bureau data); however, there is no formalized and consistently applied data quality control framework governing the ingestion and preprocessing stages. Data validation checks (e.g., missing value thresholds, outlier detection, referential integrity checks) are either absent or implemented inconsistently across different data feeds. Additionally, there is no automated reconciliation between source systems and model input datasets to ensure completeness. Data transformations applied during preprocessing are not fully documented, and there is limited evidence of controls to detect data corruption or unintended alterations during data handling. Historical data used for model development also shows gaps in key fields, with no documented imputation strategy or justification.",
     "business_justification": "Deficiencies in data quality controls increase the risk that the model is trained and executed on inaccurate, incomplete, or inconsistent data, directly affecting the reliability of model outputs. This can lead to systematic bias in risk estimates, misinformed credit decisions, and potential misstatement of risk-weighted assets. Lack of robust data governance frameworks may result in non-compliance with supervisory expectations around data integrity, increasing the likelihood of regulatory findings or capital penalties.",
     "suggested_remediation": "- Establish a formal data quality framework covering all upstream source systems with documented validation rules (missing value thresholds, outlier bounds, referential integrity checks)\n- Implement automated reconciliation controls between source systems and model input datasets, with exception reporting and escalation procedures\n- Document all data transformation and preprocessing steps, including imputation strategies and rationale for handling missing or anomalous data\n- Introduce a data lineage register to track data flows from source to model input, enabling traceability and auditability\n- Conduct a retrospective data quality assessment on historical training data and remediate identified gaps prior to next model recalibration\n- Schedule periodic data quality reviews aligned with model monitoring cycles to ensure continued integrity of model inputs"},
    {"finding_id": "F002", "model_theme": "Model Documentation",
     "title": "Insufficient Model Development Documentation and Transparency",
     "description": "The model documentation does not provide a comprehensive and traceable account of the development process. Key elements such as variable selection criteria, feature engineering steps, transformation logic, and exclusion of candidate variables are not adequately described. The rationale for selecting the final model methodology over alternative approaches is not documented, and there is limited discussion of model limitations and assumptions. Furthermore, the documentation lacks reproducibility — there is no clear linkage between documented steps and the actual codebase or datasets used.",
     "business_justification": "Inadequate documentation reduces transparency and impairs the ability of independent validation teams and auditors to effectively review and challenge the model. This increases model risk by allowing potential conceptual or technical weaknesses to remain unidentified. Additionally, poor documentation creates key-person dependency, as future redevelopment or recalibration efforts may not be feasible without significant rework. Regulatory expectations emphasize comprehensive documentation as a cornerstone of model risk management.",
     "suggested_remediation": "- Develop a comprehensive model development document (MDD) covering the full model lifecycle: data sourcing, variable selection rationale, feature engineering, methodology choice, and testing results\n- Document the exclusion rationale for all candidate variables considered but not included in the final model\n- Establish explicit linkage between the documentation and the corresponding code repository, dataset versions, and output artefacts\n- Include a dedicated section on model limitations, known weaknesses, and scenarios where the model should not be applied without overlay\n- Introduce a documentation standard or template aligned with regulatory expectations (e.g., SR 11-7) and enforce peer review prior to model approval\n- Conduct a retrospective documentation review for all models in the inventory and remediate gaps within a defined timeframe"},
    {"finding_id": "F003", "model_theme": "Model Governance",
     "title": "Unclear Model Ownership and Weak Governance Structure",
     "description": "The model governance framework does not clearly define roles and responsibilities across model development, validation, approval, and ongoing monitoring functions. There is no formally designated model owner accountable for the model's performance and compliance throughout its lifecycle. The model inventory is incomplete and does not consistently capture key attributes such as model tiering, usage, approval status, and review frequency. Escalation procedures for model-related issues are not formally documented or consistently followed.",
     "business_justification": "Weak governance structures create ambiguity in accountability, increasing the risk that model deficiencies remain unaddressed or are not escalated in a timely manner. This may lead to continued reliance on flawed models for critical business decisions. Lack of governance oversight may result in non-compliance with internal policies and regulatory expectations, potentially triggering supervisory findings and enforcement actions.",
     "suggested_remediation": "- Formally designate model owners for each model in the inventory, with documented accountability for performance, compliance, and lifecycle management\n- Define and publish a RACI matrix covering all model risk management functions: development, validation, approval, monitoring, and decommissioning\n- Complete and maintain a comprehensive model inventory capturing model tiering, intended use, approval status, last review date, and assigned owner\n- Establish and document escalation procedures for model-related issues, including trigger thresholds and governance forum routing\n- Integrate model governance activities into existing risk committee structures with regular reporting on model health and outstanding findings\n- Conduct an annual governance maturity assessment against internal policy and regulatory frameworks"},
    {"finding_id": "F004", "model_theme": "Model Methodology",
     "title": "Unjustified and Untested Model Assumptions",
     "description": "The model incorporates several key assumptions, including linear relationships between predictors and target variables, independence among explanatory variables, and stationarity of underlying data distributions. However, these assumptions have not been empirically tested or validated through statistical analysis. There is no evidence of diagnostic testing (e.g., multicollinearity checks, residual analysis, stability tests) to support the appropriateness of the chosen methodology. Alternative modeling approaches that may better capture non-linear relationships or interactions were not considered.",
     "business_justification": "Failure to validate core model assumptions undermines the conceptual soundness of the model and increases the likelihood of biased or unstable outputs. This can result in incorrect risk assessments, particularly under changing economic conditions or portfolio characteristics. Models lacking methodological rigor may be deemed unreliable by regulators, leading to increased scrutiny or disqualification for regulatory capital purposes.",
     "suggested_remediation": "- Conduct formal empirical testing of all key model assumptions, including linearity checks, independence tests (VIF for multicollinearity), stationarity tests (ADF/KPSS), and residual diagnostics\n- Document test results and provide explicit statistical justification for retaining or relaxing each assumption\n- Perform a structured benchmarking exercise evaluating at least two alternative methodologies (e.g., logistic regression vs. gradient boosting vs. scorecard) before finalising the model approach\n- Document the selection rationale with quantitative comparison of performance metrics across candidate models\n- Introduce a formal model conceptual soundness review as part of the validation sign-off process\n- Schedule periodic reassessment of assumptions as part of annual model review, particularly when portfolio composition or macro conditions change significantly"},
    {"finding_id": "F005", "model_theme": "Model Performance",
     "title": "Absence of Robust Ongoing Performance Monitoring Framework",
     "description": "Post-implementation monitoring of model performance is not supported by a formalized framework. Key performance indicators such as discriminatory power (e.g., Gini coefficient), calibration accuracy, population stability index (PSI), and characteristic stability index (CSI) are either not defined or not monitored on a periodic basis. There are no established thresholds or trigger limits to identify model deterioration. Monitoring activities, where performed, are manual and lack proper documentation, limiting traceability.",
     "business_justification": "Without systematic performance monitoring, degradation in model accuracy or stability may go undetected, leading to continued reliance on outdated or mis-specified models. This can adversely impact credit approvals, pricing decisions, and may result in financial losses. Regulators expect institutions to continuously monitor model performance and take timely corrective actions; failure increases the risk of supervisory intervention.",
     "suggested_remediation": "- Define and implement a formal model performance monitoring framework with documented KPIs including Gini/AUC, calibration ratios, PSI, and CSI\n- Establish quantitative alert thresholds and escalation triggers for each metric (e.g., PSI > 0.25 triggers immediate review)\n- Implement a periodic monitoring schedule (monthly or quarterly) calibrated to model materiality and risk tier\n- Automate the monitoring process where feasible to reduce manual effort and improve consistency and auditability\n- Introduce out-of-sample and out-of-time backtesting as a standard component of each monitoring cycle\n- Benchmark model performance against challenger models or industry benchmarks on an annual basis\n- Integrate monitoring outputs into governance forums with documented review, sign-off, and remediation tracking"},
    {"finding_id": "F006", "model_theme": "Model Implementation",
     "title": "Lack of Controls Ensuring Consistency Between Development and Production Environments",
     "description": "There are material discrepancies between the model implementation in the development environment and the version deployed in production. Differences were observed in data preprocessing logic, parameter values, and treatment of missing values. There is no formal reconciliation or validation process to ensure that the production implementation faithfully reflects the approved model. Deployment processes lack automated testing, version control, and segregation of duties, increasing the risk of unauthorized or erroneous changes.",
     "business_justification": "Implementation inconsistencies can lead to divergence between expected and actual model outputs, undermining the reliability of model-driven decisions. This introduces operational risk and may result in incorrect risk assessments or financial reporting. Lack of robust implementation controls weakens auditability and increases the likelihood of regulatory findings.",
     "suggested_remediation": "- Establish a formal pre-deployment reconciliation process to validate that production code faithfully reflects the approved model specification, including parameter values, preprocessing logic, and missing value treatment\n- Implement automated unit and integration testing covering all critical code paths prior to any production deployment\n- Introduce version control (e.g., Git) for all model code, configuration files, and associated artefacts with mandatory peer review and approval workflows\n- Enforce segregation of duties between model development, testing, and production deployment functions\n- Maintain a change log for all production modifications with documented approvals and impact assessments\n- Conduct a point-in-time reconciliation of the current production model against the approved model document and remediate any identified discrepancies"},
    {"finding_id": "F007", "model_theme": "Modelling Input Data",
     "title": "Use of Non-Representative and Outdated Training Data",
     "description": "The model has been developed using historical data that does not adequately reflect the current portfolio composition, customer behavior, or prevailing macroeconomic conditions. There is no evidence of data refresh or recalibration since initial development. Additionally, structural breaks (e.g., post-pandemic changes in borrower behavior) have not been incorporated into the dataset. Sampling techniques used during model development are not documented, raising concerns about potential selection bias.",
     "business_justification": "Use of non-representative data can significantly impair model accuracy and lead to biased predictions, particularly in dynamic environments. This may result in underestimation or overestimation of risk, affecting capital adequacy and strategic decision-making. Regulators expect models to be based on relevant and representative data; failure may lead to model rejection or capital penalties.",
     "suggested_remediation": "- Conduct a representativeness assessment comparing the training dataset composition against the current live portfolio across key dimensions (product type, vintage, geography, credit tier)\n- Refresh the training dataset to incorporate recent origination cohorts and post-structural-break observations; document the rationale for the selected observation period\n- Document all sampling techniques applied during development, including stratification, exclusions, and weighting, and provide statistical justification\n- Establish a data refresh policy specifying minimum refresh frequency (e.g., annual) and trigger-based refresh criteria linked to portfolio drift indicators\n- Perform sensitivity analysis to quantify the performance impact of using updated vs. historical data, and recalibrate the model if material degradation is identified\n- Include representativeness testing as a standard component of model validation and periodic review"},
    {"finding_id": "F008", "model_theme": "Model Documentation",
     "title": "Incomplete Documentation of Model Limitations and Use Constraints",
     "description": "The model documentation does not clearly articulate the known limitations, assumptions, and appropriate use cases of the model. There is no guidance on scenarios where the model may produce unreliable results or should not be used. Additionally, there is no documentation of compensating controls or overlays required when limitations are triggered.",
     "business_justification": "Lack of clarity on model limitations increases the risk of misuse or over-reliance in inappropriate contexts. This can lead to incorrect decisions and potential financial losses. Proper documentation of limitations is a key regulatory expectation to ensure responsible model usage.",
     "suggested_remediation": "- Add a dedicated 'Model Limitations and Use Constraints' section to the model development document, covering known weaknesses, boundary conditions, and prohibited use cases\n- Document all scenarios where expert overlays or compensating controls are required, including the trigger conditions and approved adjustment ranges\n- Establish a formal model use policy requiring sign-off from the model owner before applying the model in new contexts or products\n- Circulate the limitations register to all model users and relevant business stakeholders, with acknowledgement of receipt\n- Review and update the limitations documentation at each annual model review or following any material change to the model or its operating environment"},
    {"finding_id": "F009", "model_theme": "Model Governance",
     "title": "Inadequate Model Lifecycle Management and Review Process",
     "description": "The institution does not maintain a structured model lifecycle management framework covering development, validation, approval, implementation, monitoring, and decommissioning stages. Periodic model reviews are not conducted consistently, and there is no defined frequency based on model risk tiering.",
     "business_justification": "Weak lifecycle management increases the likelihood that outdated or high-risk models remain in use without proper oversight, leading to elevated model risk and potential regulatory non-compliance.",
     "suggested_remediation": "- Define and implement a formal model lifecycle management framework with documented stage gates for development, validation, approval, deployment, monitoring, and retirement\n- Establish risk-tiered review frequencies (e.g., Tier 1: annual full review; Tier 2: biennial; Tier 3: triennial) and enforce compliance through the model inventory\n- Assign lifecycle management responsibilities to named model owners and document the process in the model risk management policy\n- Implement a model inventory dashboard tracking lifecycle stage, last review date, next due date, and outstanding actions for each model\n- Introduce a formal decommissioning process requiring documentation of the rationale, migration path, and approval before a model is retired"},
    {"finding_id": "F010", "model_theme": "Model Methodology",
     "title": "Limited Consideration of Alternative Modeling Approaches",
     "description": "The model development process did not include a comprehensive benchmarking exercise against alternative methodologies. There is no evidence that more advanced or appropriate techniques were evaluated before finalizing the model.",
     "business_justification": "Failure to consider alternative approaches may result in suboptimal model performance and missed opportunities to improve predictive accuracy, ultimately impacting business outcomes and regulatory standing.",
     "suggested_remediation": "- Conduct a structured methodology selection exercise evaluating a minimum of three candidate approaches (e.g., logistic regression, gradient boosting, scorecard) using consistent train/test datasets\n- Document quantitative performance comparisons across candidates using standardised metrics (Gini, AUC, KS statistic, calibration error)\n- Provide a written rationale for the chosen methodology, explicitly addressing why alternatives were rejected or deferred\n- Incorporate methodology benchmarking as a mandatory step in the model development standard and validation checklist\n- Revisit the benchmarking analysis at each major model review to assess whether newer techniques have become more appropriate given portfolio evolution"},
    {"finding_id": "F011", "model_theme": "Model Performance",
     "title": "Lack of Backtesting and Benchmarking Analysis",
     "description": "The model has not been subjected to rigorous backtesting using out-of-sample data or benchmarking against challenger models. There is insufficient evidence of discriminatory power or calibration assessments on holdout datasets.",
     "business_justification": "Absence of backtesting limits confidence in the model's predictive power and stability, increasing the risk of relying on an underperforming model for critical business and regulatory decisions.",
     "suggested_remediation": "- Conduct an immediate out-of-sample and out-of-time backtesting exercise using held-out data, reporting Gini, AUC, KS statistic, and calibration metrics\n- Establish a challenger model or benchmark comparison using at least one alternative methodology to assess relative predictive performance\n- Formalise backtesting as a recurring activity within the monitoring calendar, with minimum frequency aligned to model risk tier\n- Document all backtesting methodologies, datasets, and results in a standardised report for governance review and audit trail\n- Define threshold-based action triggers (e.g., Gini decline > 5 percentage points triggers escalation) and link results to the model redevelopment or recalibration workflow"},
    {"finding_id": "F012", "model_theme": "Model Implementation",
     "title": "Absence of Robust Version Control and Change Tracking",
     "description": "Model code and associated artifacts are not maintained in a controlled versioning system. Changes are not systematically logged, reviewed, or approved. There is no audit trail of modifications to model logic or parameters.",
     "business_justification": "Lack of version control increases operational risk and reduces traceability, making it difficult to investigate issues or demonstrate compliance during audits and regulatory reviews.",
     "suggested_remediation": "- Implement a version control system (e.g., Git with a protected main branch) for all model code, configuration files, and associated artefacts with mandatory commit messages and pull request reviews\n- Enforce a change management policy requiring documented approval for all modifications to model logic, parameters, or preprocessing steps\n- Maintain a change log capturing what was changed, by whom, when, and with what approval — linked to the relevant governance decision\n- Introduce automated CI/CD pipeline checks that prevent unapproved code from being promoted to production environments\n- Conduct a retrospective audit of recent production changes and remediate any that lack documentation or approval records\n- Provide training to model development and implementation teams on version control standards and change management obligations"},
]

for f in SAMPLE_FINDINGS:
    findings_db.append(f)
run_clustering()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
