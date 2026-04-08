
import numpy as np
import faiss


def _to_f32(arr: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(arr, dtype=np.float32)


def build_index(train: np.ndarray, normalize: bool = False) -> faiss.Index:
    train = _to_f32(train)
    if normalize:
        train = train.copy()
        faiss.normalize_L2(train)
    D = train.shape[1]
    index = faiss.IndexFlatL2(D)
    try:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    except Exception:
        pass  # no GPU available, stay on CPU
    index.add(train)
    return index


def query_index(
    index: faiss.Index,
    query: np.ndarray,
    k: int,
    normalize: bool = False,
    batch_size: int = 65536,
) -> np.ndarray:
    query = _to_f32(query)
    if normalize:
        query = query.copy()
        faiss.normalize_L2(query)
    k = min(k, index.ntotal)
    results = []
    for start in range(0, len(query), batch_size):
        chunk = query[start : start + batch_size]
        distances, _ = index.search(chunk, k)
        results.append(distances)
    distances = np.concatenate(results, axis=0)
    return np.sqrt(np.clip(distances, 0.0, None))


def faiss_knn(
    train: np.ndarray,
    query: np.ndarray,
    k: int,
    normalize: bool = False,
) -> np.ndarray:
    """Build index and query in one call; returns distances of shape (N_query, k).

    Prefer build_index + query_index when the same index is queried more than once.
    """
    index = build_index(train, normalize=normalize)
    return query_index(index, query, k, normalize=normalize)