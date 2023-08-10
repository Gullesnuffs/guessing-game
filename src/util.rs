pub fn subsets_of_size(n: usize, k: usize) -> Vec<Vec<bool>> {
    if k == 0 {
        return vec![vec![false; n]];
    }
    if n == 0 {
        return vec![];
    }
    let mut result = subsets_of_size(n - 1, k)
        .into_iter()
        .map(|mut subset| {
            subset.push(false);
            subset
        })
        .collect::<Vec<_>>();
    for mut subset in subsets_of_size(n - 1, k - 1) {
        subset.push(true);
        result.push(subset);
    }
    result
}