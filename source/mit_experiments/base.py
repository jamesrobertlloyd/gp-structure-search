import mit_job_controller as mjc
import mitparallel as mp



def remove_duplicates(kernels, X, n_eval=250, local_computation=True):
    '''
    Test the top n_eval performing kernels for equivalence, in terms of their covariance matrix evaluated on training inputs
    Assumes kernels is a list of ScoredKernel objects
    '''
    #### HACK - this needs lots of computing power - do it locally with multi-threading
    local_computation = True
    # Sort
    kernels = sorted(kernels, key=ScoredKernel.score, reverse=False)
    # Find covariance similarity for top n_eval
    n_eval = min(n_eval, len(kernels))
    similarity_matrix = covariance_similarity(kernels[:n_eval], X, local_computation=local_computation)
    # Remove similar kernels
    #### TODO - What is a good heuristic for determining equivalence?
    ####      - Currently using Frobenius norm - truncate if Frobenius norm is below 1% of average
    cut_off = similarity_matrix.mean() / 100.0
    equivalence_graph = similarity_matrix < cut_off
    # For all kernels (best first)
    for i in range(n_eval):
        # For all other worse kernels
        for j in range(i+1, n_eval):
            # If equivalent
            if equivalence_graph[i,j]:
                # Destroy the inferior duplicate
                kernels[j] = None
    # Sort the results
    kernels = [k for k in kernels if k is not None]
    kernels = sorted(kernels, key=ScoredKernel.score, reverse=True)
    return kernels



