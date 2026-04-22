from flask import Flask, render_template, request, jsonify
import json
import re
import math
import threading
import io
from collections import defaultdict

# Pre-load heavy deps at startup so first upload is instant
try:
    import pandas as pd
    import openpyxl  # noqa: F401 – ensures xlsx engine is available
    _PANDAS_OK = True
except ImportError:
    _PANDAS_OK = False

app = Flask(__name__, template_folder='.', static_folder='.', static_url_path='/')

# ── Global clustering state ───────────────────────────────────────────────────
_cluster_lock = threading.Lock()
_cluster_thread = None
_clustering_busy = False   # True while a background cluster job is running


# ── Flask error handlers – always return JSON, never HTML ────────────────────
@app.errorhandler(400)
@app.errorhandler(404)
@app.errorhandler(405)
@app.errorhandler(500)
def json_error(e):
    code = getattr(e, 'code', 500)
    return jsonify({'error': str(e)}), code

# ── In-memory store ───────────────────────────────────────────────────────────
findings_db = []
clusters_db = []

# ── Stop-words: generic English + MRM domain noise ───────────────────────────
# Words that appear in virtually every MRM finding are useless for clustering —
# they inflate frequency counts without adding discriminative signal.
# "model", "risk", "finding", "data", "process" etc. belong in this list.
# ── N-gram bigram extractor ───────────────────────────────────────────────────
# Bigrams dramatically improve semantic coherence — "performance monitoring"
# is far more discriminative than "performance" or "monitoring" alone.
def bigrams(tokens: list) -> list:
    return [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)]

STOP_WORDS = {
    # English function words
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
    'when','how','well','due','per','has','been','provide','provided',
    'identify','identified','identify','identified','address','addressed',
    'conduct','conducted','establish','established','implement','implemented',
    'define','defined','review','reviewed','require','required','include',
    'included','use','used','using','allow','allowed','ensure','ensure',
    # MRM domain noise — present in virtually every finding, zero discriminative power
    'model','models','risk','risks','finding','findings','data','process',
    'processes','control','controls','management','framework','system',
    'systems','current','existing','new','high','low','key','major',
    'significant','appropriate','adequate','inadequate','insufficient',
    'formal','informal','internal','external','related','relevant',
    'specific','general','overall','potential','possible','likely',
    'number','level','levels','area','areas','issue','issues','concern',
    'concerns','requirement','requirements','standard','standards',
    'policy','policies','procedure','procedures','practice','practices',
    'activity','activities','report','reports','reporting','evidence',
    'information','institution','team','teams','function','functions',
}

# ── MRM theme → canonical label (strips the word "model" prefix) ─────────────
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


# ── Tokenizer ─────────────────────────────────────────────────────────────────
def tokenize(text: str) -> list:
    """Lowercase, extract alpha tokens >=3 chars, remove stop-words."""
    tokens = re.findall(r'\b[a-z][a-z\-]{2,}\b', text.lower())
    return [t for t in tokens if t not in STOP_WORDS]


def tokenize_with_bigrams(text: str) -> list:
    """Return unigrams + bigrams -- richer semantic signal."""
    uni = tokenize(text)
    bi = [f"{uni[i]}_{uni[i+1]}" for i in range(len(uni) - 1)]
    return uni + bi


# ── Weighted field tokenizer ──────────────────────────────────────────────────
# Field weights reflect actual semantic density of each field:
#   description         — the richest source of what the finding IS about;
#                         long, specific, full of domain vocabulary → highest weight
#   business_justification — explains WHY it matters; regulatory language,
#                         risk framing, consequence vocabulary → high weight
#   title               — concise summary but often generic phrasing; useful
#                         but should not dominate over the full-text fields
#   suggested_remediation — action verbs, process steps; weakest discriminator
#
# model_theme is NOT used as a clustering signal at all — clustering is driven
# purely by the textual content of the four fields above.  Theme is only used
# for post-hoc label generation and k-floor anchoring.
FIELD_WEIGHTS = {
    'description':            6,   # Primary semantic signal — richest content
    'business_justification': 5,   # Risk framing, regulatory context — high signal
    'title':                  2,   # Useful hint but often generic; keep low
    'suggested_remediation':  1,   # Action language; weakest discriminator
}


def weighted_token_counts(finding: dict) -> dict:
    """
    Return {token: weighted_count} for a single finding.
    Includes unigrams + bigrams from all text fields.
    model_theme is NOT used here — clustering is purely content-driven.
    """
    counts = defaultdict(float)
    for field, weight in FIELD_WEIGHTS.items():
        val = finding.get(field, '') or ''
        for tok in tokenize_with_bigrams(val):
            counts[tok] += weight
    return counts


# ── TF-IDF vectoriser (sparse, sublinear TF) ──────────────────────────────────
def build_tfidf(token_counts_list: list):
    """
    Build normalised TF-IDF vectors from pre-computed weighted token counts.

    Uses sublinear TF (1 + log(tf)) to dampen term frequency dominance.
    IDF uses smoothed formula: log((N+1)/(df+1)) + 1.
    Returns: (matrix as list-of-lists, vocab list, word→index dict)
    """
    N = len(token_counts_list)
    # Document frequency
    df = defaultdict(int)
    for tc in token_counts_list:
        for t in tc:
            df[t] += 1

    # Only keep terms that appear in at least 1 doc but fewer than 95% of docs
    # (near-universal terms are noise even after stop-word removal)
    max_df = max(1, int(0.95 * N))
    vocab = [t for t, d in df.items() if 1 <= d <= max_df]
    word2idx = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)

    vectors = []
    for tc in token_counts_list:
        vec = [0.0] * V
        for t, wcount in tc.items():
            idx = word2idx.get(t)
            if idx is None:
                continue
            tf = 1.0 + math.log(wcount) if wcount > 0 else 0.0
            idf = math.log((N + 1) / (df[t] + 1)) + 1.0
            vec[idx] = tf * idf
        # L2-normalise
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        vectors.append([x / norm for x in vec])
    return vectors, vocab, word2idx


# ── Fast dot-product (cosine on L2-normalised vectors = dot product) ──────────
def dot(a: list, b: list) -> float:
    return sum(a[i] * b[i] for i in range(len(a)))


# ── Matrix-vector multiply: all cosines at once ───────────────────────────────
def cosine_all(matrix: list, vec: list) -> list:
    """Return cosine similarity of vec against every row in matrix."""
    return [dot(row, vec) for row in matrix]


# ── K-Means++ with sparse centroid updates ────────────────────────────────────
def kmeans(vectors: list, k: int, max_iter: int = 100, n_restarts: int = 3, seed: int = 42):
    import random
    n = len(vectors)
    if n == 0:
        return [], []
    k = min(k, n)
    V = len(vectors[0])

    best_labels, best_inertia = None, float('inf')

    for restart in range(n_restarts):
        rng = random.Random(seed + restart * 31)

        # K-Means++ init
        first = rng.randint(0, n - 1)
        centroids = [vectors[first][:]]
        for _ in range(k - 1):
            sims = [max(dot(v, c) for c in centroids) for v in vectors]
            # distance = 1 - max_sim; sample proportional to distance²
            dists = [max(0.0, 1.0 - s) ** 2 for s in sims]
            total = sum(dists) or 1.0
            r, cum = rng.random() * total, 0.0
            chosen = n - 1
            for i, d in enumerate(dists):
                cum += d
                if cum >= r:
                    chosen = i
                    break
            centroids.append(vectors[chosen][:])

        labels = [0] * n
        for _ in range(max_iter):
            # Assign: for each point find closest centroid
            new_labels = []
            for v in vectors:
                sims = cosine_all(centroids, v)
                new_labels.append(sims.index(max(sims)))
            if new_labels == labels:
                break
            labels = new_labels

            # Update centroids
            for j in range(k):
                members = [vectors[i] for i, lbl in enumerate(labels) if lbl == j]
                if not members:
                    # Empty cluster: reinit to farthest point
                    farthest = max(range(n), key=lambda i: 1.0 - dot(vectors[i], centroids[labels[i]]))
                    centroids[j] = vectors[farthest][:]
                    continue
                c = [sum(members[m][d] for m in range(len(members))) / len(members) for d in range(V)]
                norm = math.sqrt(sum(x * x for x in c)) or 1.0
                centroids[j] = [x / norm for x in c]

        inertia = sum(1.0 - dot(vectors[i], centroids[labels[i]]) for i in range(n))
        if inertia < best_inertia:
            best_inertia, best_labels = inertia, labels[:]

    return best_labels, None


# ── Bisecting K-Means ─────────────────────────────────────────────────────────
def bisecting_kmeans(vectors: list, k: int, seed: int = 42):
    """
    Repeatedly bisect the highest-inertia cluster until we have k clusters.
    More stable than flat K-Means for larger k; avoids bad local minima.
    """
    n = len(vectors)
    if n == 0:
        return [], []
    k = min(k, n)
    V = len(vectors[0])

    clusters = [list(range(n))]

    def inertia_of(idx_list):
        if len(idx_list) <= 1:
            return 0.0
        vecs = [vectors[i] for i in idx_list]
        c = [sum(v[d] for v in vecs) / len(vecs) for d in range(V)]
        norm = math.sqrt(sum(x * x for x in c)) or 1.0
        c = [x / norm for x in c]
        return sum(1.0 - dot(vectors[i], c) for i in idx_list)

    while len(clusters) < k:
        # Pick cluster with highest inertia (most room to improve)
        target_idx = max(range(len(clusters)), key=lambda j: inertia_of(clusters[j]))
        target = clusters[target_idx]
        if len(target) < 2:
            break

        sub_vecs = [vectors[i] for i in target]
        lbls, _ = kmeans(sub_vecs, 2, max_iter=50, n_restarts=2, seed=seed + len(clusters) * 17)
        if not lbls:
            break

        group_a = [target[i] for i, l in enumerate(lbls) if l == 0]
        group_b = [target[i] for i, l in enumerate(lbls) if l == 1]
        if not group_a or not group_b:
            break

        clusters.pop(target_idx)
        clusters.extend([group_a, group_b])

    labels = [0] * n
    for lbl, idx_list in enumerate(clusters):
        for i in idx_list:
            labels[i] = lbl
    return labels, clusters


# ── Silhouette score (sampled for speed on large datasets) ────────────────────
def _silhouette_score(vectors: list, labels: list, k: int, max_sample: int = 200) -> float:
    """
    Mean silhouette coefficient.  Samples up to max_sample points to keep
    O(n²) cost manageable — accurate enough for k-selection.
    """
    import random
    n = len(vectors)
    if k <= 1 or n <= k:
        return -1.0

    cluster_members = defaultdict(list)
    for i, l in enumerate(labels):
        cluster_members[l].append(i)

    # Sample indices for scoring
    indices = list(range(n))
    if n > max_sample:
        random.seed(7)
        indices = random.sample(indices, max_sample)

    scores = []
    for i in indices:
        lbl = labels[i]
        same = [j for j in cluster_members[lbl] if j != i]
        if not same:
            scores.append(0.0)
            continue
        a = sum(1.0 - dot(vectors[i], vectors[j]) for j in same) / len(same)
        b_vals = [
            sum(1.0 - dot(vectors[i], vectors[j]) for j in members) / len(members)
            for other_lbl, members in cluster_members.items()
            if other_lbl != lbl and members
        ]
        b = min(b_vals) if b_vals else 0.0
        denom = max(a, b)
        scores.append((b - a) / denom if denom else 0.0)

    return sum(scores) / len(scores) if scores else -1.0


# ---- Stable k-selection: theme-anchored with silhouette confirmation --------
def _theme_purity(vectors: list, labels: list, findings: list) -> float:
    """
    Fraction of findings that share their cluster's dominant theme.
    A pure cluster has all findings from the same theme.
    Higher is better (max 1.0).
    """
    cluster_themes = defaultdict(list)
    for i, (lbl, f) in enumerate(zip(labels, findings)):
        t = normalize_theme(f.get('model_theme', ''))
        cluster_themes[lbl].append(t)
    purity = 0.0
    for lbl, themes in cluster_themes.items():
        if not themes:
            continue
        dominant_count = max(themes.count(t) for t in set(themes)) if themes else 0
        purity += dominant_count
    return purity / len(findings) if findings else 0.0


def _auto_select_k(vectors: list, n: int, theme_count: int, findings: list = None) -> int:
    """
    Stable k-selection using a composite score:
      score = 0.5 * silhouette + 0.5 * theme_purity - 0.003 * k

    The theme-purity term anchors k near the number of distinct themes,
    preventing the cluster count from changing when new findings are added
    (as long as they belong to existing themes).

    k_floor = max(theme_count, 2) ensures we never produce fewer clusters
    than there are distinct themes in the corpus.
    """
    if n <= 2:
        return 1
    if n <= 3:
        return 2

    # Hard floor: never fewer clusters than distinct themes (semantic anchoring)
    k_floor = max(2, theme_count) if theme_count else 2

    if n <= 10:
        k_min, k_max = k_floor, min(n - 1, max(k_floor + 2, 5))
    elif n <= 30:
        k_min, k_max = k_floor, min(n // 2, max(k_floor + 3, 8))
    elif n <= 100:
        k_min, k_max = k_floor, min(n // 4, max(k_floor + 4, 12))
    else:
        k_min, k_max = k_floor, min(n // 8, max(k_floor + 5, 20))

    k_max = max(k_max, k_min + 1)

    best_k, best_score = k_min, -float('inf')
    no_improve = 0
    prev_clusters = [list(range(n))]

    for k in range(k_min, k_max + 1):
        # Grow cluster list incrementally (reuse previous work)
        while len(prev_clusters) < k:
            V = len(vectors[0])

            def inertia_of(idx_list, _V=V):
                if len(idx_list) <= 1:
                    return 0.0
                vecs = [vectors[i] for i in idx_list]
                c = [sum(v[d] for v in vecs) / len(vecs) for d in range(_V)]
                norm = math.sqrt(sum(x * x for x in c)) or 1.0
                c = [x / norm for x in c]
                return sum(1.0 - dot(vectors[i], c) for i in idx_list)

            target_idx = max(range(len(prev_clusters)),
                             key=lambda j: inertia_of(prev_clusters[j]))
            target = prev_clusters[target_idx]
            if len(target) < 2:
                break
            sub_vecs = [vectors[i] for i in target]
            lbls, _ = kmeans(sub_vecs, 2, max_iter=50, n_restarts=2,
                             seed=42 + len(prev_clusters) * 17)
            if not lbls:
                break
            ga = [target[i] for i, l in enumerate(lbls) if l == 0]
            gb = [target[i] for i, l in enumerate(lbls) if l == 1]
            if not ga or not gb:
                break
            prev_clusters.pop(target_idx)
            prev_clusters.extend([ga, gb])

        if len(prev_clusters) < k:
            break

        labels = [0] * n
        for lbl, idx_list in enumerate(prev_clusters):
            for i in idx_list:
                labels[i] = lbl

        sil = _silhouette_score(vectors, labels, k)
        purity = _theme_purity(vectors, labels, findings or []) if findings else 0.5
        # Composite: silhouette measures geometric quality, purity measures
        # semantic coherence; tiny penalty discourages unnecessary splits
        score = 0.50 * sil + 0.50 * purity - 0.003 * k

        if score > best_score + 0.003:
            best_score, best_k = score, k
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= 3:
                break

    return best_k


# ── Semantic cluster label: dominant theme + top discriminative bigrams ────────
def _cluster_label(fids: list, all_clusters_fids: list) -> str:
    """
    Generate a human-readable semantic label for a cluster.
    Returns: "<DominantTheme>: <top bigram phrase>, <second bigram phrase>"
    """
    fid_set = {f['finding_id']: f for f in findings_db}
    # Dominant theme
    themes = []
    for fid in fids:
        f = fid_set.get(fid)
        if f and f.get('model_theme'):
            themes.append(normalize_theme(f['model_theme']))
    dominant_theme = ''
    if themes:
        dominant_theme = max(set(themes), key=themes.count)
        dominant_theme = dominant_theme.replace('_', ' ').title()

    # Top discriminative bigrams (prefer bigrams over unigrams for readability)
    freq_in = defaultdict(float)
    for fid in fids:
        f = fid_set.get(fid)
        if not f:
            continue
        tc = weighted_token_counts(f)
        for tok, wc in tc.items():
            if '_' in tok and not tok.endswith('_seed'):   # bigrams only
                freq_in[tok] += wc

    # IDF-weight against other clusters to find discriminative bigrams
    all_fids_flat = [fid for cluster in all_clusters_fids for fid in cluster]
    freq_all = defaultdict(float)
    for fid in all_fids_flat:
        f = fid_set.get(fid)
        if not f:
            continue
        tc = weighted_token_counts(f)
        for tok, wc in tc.items():
            if '_' in tok and not tok.endswith('_seed'):
                freq_all[tok] += wc

    n_clusters = max(1, len(all_clusters_fids))
    tfidf_bigrams = {}
    for tok, freq in freq_in.items():
        # Simple TF-IDF: normalize by cluster size, penalize globally common terms
        tf = freq / max(1, len(fids))
        idf = math.log(n_clusters / max(1, sum(1 for cl in all_clusters_fids
                                                if any(weighted_token_counts(fid_set.get(fid, {})).get(tok, 0) > 0
                                                       for fid in cl))))
        tfidf_bigrams[tok] = tf * idf

    top_bi = sorted(tfidf_bigrams, key=lambda x: -tfidf_bigrams[x])[:2]
    phrase = ', '.join(t.replace('_', ' ') for t in top_bi)

    if dominant_theme and phrase:
        return f"{dominant_theme}: {phrase}"
    elif dominant_theme:
        return dominant_theme
    return phrase or 'Mixed'


def _top_signals(fids: list) -> str:
    """
    Return top 12 discriminative tokens (bigrams preferred) for a cluster,
    excluding seed tokens and common noise.
    """
    freq = defaultdict(float)
    fid_set = {f['finding_id']: f for f in findings_db}
    for fid in fids:
        f = fid_set.get(fid)
        if not f:
            continue
        tc = weighted_token_counts(f)
        for tok, wc in tc.items():
            if not tok.endswith('_seed'):
                freq[tok] += wc
    # Prefer bigrams (more semantic), then unigrams
    bigram_tokens = sorted(
        [t for t in freq if '_' in t], key=lambda x: -freq[x])[:6]
    unigram_tokens = sorted(
        [t for t in freq if '_' not in t], key=lambda x: -freq[x])[:6]
    top = bigram_tokens + [u for u in unigram_tokens if u not in bigram_tokens]
    return ', '.join(t.replace('_', ' ') for t in top[:12])


def _generate_why(fids: list, label: str = '') -> str:
    themes = []
    fid_set = {f['finding_id']: f for f in findings_db}
    for fid in fids:
        f = fid_set.get(fid)
        if f and f.get('model_theme'):
            themes.append(f['model_theme'])
    theme_str = ''
    if themes:
        unique = list(dict.fromkeys(themes))
        theme_str = f" under the theme(s): {', '.join(unique)}."
    label_str = f" Semantic label: '{label}'." if label else ''
    return (
        f"Findings {', '.join(fids)} share overlapping vocabulary and risk focus"
        f"{theme_str}{label_str} Bisecting K-Means on sublinear TF-IDF vectors "
        "(description×6 + business justification×5 + title×2 + remediation×1, "
        "with bigrams) detected strong semantic similarity purely from finding content — "
        "model_theme is not used as a clustering signal."
    )


def _do_clustering(k: int = None):
    """Actual clustering work – runs in a background thread."""
    global clusters_db, _clustering_busy
    _clustering_busy = True
    try:
        if not findings_db:
            with _cluster_lock:
                clusters_db = []
            return

        n = len(findings_db)

        # Build per-finding token→weighted_count dicts (model_theme excluded)
        token_counts = [weighted_token_counts(f) for f in findings_db]
        vectors, vocab, word2idx = build_tfidf(token_counts)

        # Count distinct canonical themes as a k hint
        theme_set = set()
        for f in findings_db:
            t = normalize_theme(f.get('model_theme', ''))
            if t:
                theme_set.add(t)
        theme_count = len(theme_set)

        if k is None:
            k = _auto_select_k(vectors, n, theme_count, findings_db)
        k = min(k, n)

        # Bisecting K-Means — stable, avoids local minima for larger k
        labels, _ = bisecting_kmeans(vectors, k)
        if not labels:
            labels, _ = kmeans(vectors, k)

        cluster_map = defaultdict(list)
        for i, label in enumerate(labels):
            cluster_map[label].append(findings_db[i]['finding_id'])

        # Sort clusters by dominant MRM theme for stable, readable labelling
        theme_order = ['data_quality', 'documentation', 'governance',
                       'methodology', 'performance_monitoring', 'implementation']
        fid_lookup = {f['finding_id']: f for f in findings_db}

        def theme_sort_key(fids):
            counts = defaultdict(int)
            for fid in fids:
                f = fid_lookup.get(fid)
                if f:
                    counts[normalize_theme(f.get('model_theme', ''))] += 1
            dominant = max(counts, key=counts.get) if counts else ''
            try:
                return theme_order.index(dominant)
            except ValueError:
                return 99

        sorted_clusters = sorted(cluster_map.values(), key=theme_sort_key)
        all_clusters_fids = list(sorted_clusters)

        new_clusters = []
        for idx, fids in enumerate(sorted_clusters):
            label = _cluster_label(fids, all_clusters_fids)
            new_clusters.append({
                'cluster_id': f"C{idx + 1}",
                'findings_included': fids,
                'why_grouped': _generate_why(fids, label),
                'semantic_signals': _top_signals(fids),
                'semantic_label': label,
                'size': len(fids),
            })

        with _cluster_lock:
            clusters_db[:] = new_clusters
    finally:
        _clustering_busy = False


def run_clustering(k: int = None, background: bool = True):
    """
    Kick off clustering. By default runs in a daemon background thread so
    upload/add routes return instantly. Pass background=False for the
    initial seed load at startup (blocking is fine there).
    """
    global _cluster_thread
    if not background:
        _do_clustering(k)
        return
    t = threading.Thread(target=_do_clustering, args=(k,), daemon=True)
    _cluster_thread = t
    t.start()


def search_query(query: str) -> dict:
    if not clusters_db or not findings_db:
        return {'cluster': None, 'findings': [], 'score': 0}

    # Vectorise findings + query together so IDF is computed over the same corpus
    query_tc = {tok: 1.0 for tok in tokenize(query)}
    all_tcs = [weighted_token_counts(f) for f in findings_db] + [query_tc]
    vectors, _, _ = build_tfidf(all_tcs)
    q_vec = vectors[-1]
    doc_vecs = vectors[:-1]

    sims = [(dot(q_vec, dv), i) for i, dv in enumerate(doc_vecs)]
    sims.sort(reverse=True)
    top_findings = [findings_db[i] for _, i in sims[:5]]
    top_ids = {f['finding_id'] for f in top_findings}

    best_cluster, best_count = None, 0
    for c in clusters_db:
        overlap = len(set(c['findings_included']) & top_ids)
        if overlap > best_count:
            best_count, best_cluster = overlap, c

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
    if not _PANDAS_OK:
        return jsonify({'error': 'pandas/openpyxl not installed on this server'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    filename = (file.filename or '').lower()

    try:
        raw = file.read()
        if filename.endswith('.xlsx') or filename.endswith('.xls'):
            df = pd.read_excel(io.BytesIO(raw))
        elif filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(raw))
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


@app.route('/api/clusters/status', methods=['GET'])
def clusters_status():
    """Lightweight poll endpoint — returns whether clustering is still running."""
    return jsonify({'busy': _clustering_busy, 'count': len(clusters_db)})


@app.route('/api/clusters/rerun', methods=['POST'])
def rerun_clustering():
    k = request.json.get('k') if request.json else None
    # User explicitly triggered re-cluster – run synchronously so the
    # response reflects the updated cluster count.
    run_clustering(k, background=False)
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
# Blocking at startup – we want clusters ready before first request arrives
run_clustering(background=False)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
