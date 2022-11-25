use std::collections::HashSet;

// Find a pair with a given sum in an array
fn find_pair_with_given_sum(vec: &Vec<i32>, sum: i32) -> Option<(i32, i32)> {
    let mut set = HashSet::new();

    for e in vec {
        let another = sum - e;
        match set.get(&another) {
            Some(_) => return Some((*e, another)),
            _ => {
                set.insert(e);
            }
        }
    }

    return None;
}

// Check if a subarray with 0 sum exists or not
fn check_if_subarray_with_0_exists(vec: &Vec<i32>) -> bool {
    if vec.is_empty() {
        return false;
    }

    if vec[0] == 0 {
        return true;
    }

    let mut set = HashSet::new();
    let mut sum = 0;
    for e in vec {
        sum = sum + e;
        if set.contains(&sum) {
            return true;
        } else {
            set.insert(sum);
        }
    }

    return false;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_pair_with_given_sum() {
        assert_eq!(
            find_pair_with_given_sum(&vec![8, 7, 2, 5, 3, 1], 10),
            Some((2, 8))
        );
        assert_eq!(
            find_pair_with_given_sum(&vec![5, 2, 6, 8, 6, 9], 12),
            Some((6, 6))
        );
        assert_eq!(find_pair_with_given_sum(&vec![5, 2, 6, 8, 1, 9], 12), None);
        assert_eq!(find_pair_with_given_sum(&vec![], 10), None);
    }

    #[test]
    fn test_check_if_subarray_with_0_exists() {
        assert_eq!(
            check_if_subarray_with_0_exists(&vec![3, 4, -7, 3, 1, 3, 1, -4, -2, -2]),
            true
        );
        assert_eq!(check_if_subarray_with_0_exists(&vec![0, 1, 2]), true);
        assert_eq!(check_if_subarray_with_0_exists(&vec![0]), true);
        assert_eq!(check_if_subarray_with_0_exists(&vec![3, 1, -7, 2]), false);
        assert_eq!(check_if_subarray_with_0_exists(&vec![3]), false);
        assert_eq!(check_if_subarray_with_0_exists(&vec![]), false);
    }
}
