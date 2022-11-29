use std::collections::HashMap;
use std::collections::HashSet;

// Find a pair with a given sum in a vector
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

// Check if a slice with 0 sum exists or not
fn check_if_slice_with_0_exists(vec: &Vec<i32>) -> bool {
    let mut set = HashSet::from([0]);
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

// Find a maximum product of two numbers in a vector
fn find_max_product_of_2_numbers(vec: &Vec<i32>) -> i32 {
    let mut max_pos_1 = 0;
    let mut max_pos_2 = 0;
    let mut min_neg_1 = 0;
    let mut min_neg_2 = 0;

    for e in vec {
        if *e > 0 && *e > max_pos_1 {
            max_pos_2 = max_pos_1;
            max_pos_1 = *e;
        } else if *e > 0 && *e > max_pos_2 {
            max_pos_2 = *e;
        }

        if *e < 0 && *e < min_neg_1 {
            min_neg_2 = min_neg_1;
            min_neg_1 = *e;
        } else if *e < 0 && *e < min_neg_2 {
            min_neg_2 = *e;
        }
    }

    let product_of_max_positives = max_pos_1 * max_pos_2;
    let product_of_min_negatives = min_neg_1 * min_neg_2;

    match (product_of_max_positives, product_of_min_negatives) {
        (pos, neg) if pos != 0 && neg != 0 && pos >= neg => pos, //product of positives is larger
        (pos, neg) if pos != 0 && neg != 0 && pos <= neg => neg, //product of negatives is larger
        (pos, 0) if pos != 0 => pos,                             //has only a product of positives
        (0, neg) if neg != 0 => neg,                             //has only a product of negatives
        _ if max_pos_1 != 0 && min_neg_1 != 0 && vec.len() == 2 => max_pos_1 * min_neg_1, //has 1 positive & 1 negative number only
        _ => 0, //otherwise returns 0
    }
}

// Find maximum length slice with a given sum
fn find_max_length_slice_with_given_sum(vec: &Vec<i32>, sum: i32) -> Option<&[i32]> {
    let mut map: HashMap<i32, usize> = HashMap::from([(0, 0)]);

    let mut max_length: usize = 0;
    let mut max_length_start_index: Option<usize> = None;
    let mut max_length_end_index: Option<usize> = None;
    let mut current_sum: i32 = 0;
    for (index, elem) in vec.iter().enumerate() {
        current_sum += *elem;

        let sum_to_check = current_sum - sum;
        if let Some(start_position) = map.get(&sum_to_check) {
            let length: usize = index - start_position + 1;
            if length > max_length {
                max_length = length;
                max_length_start_index = Some(*start_position);
                max_length_end_index = Some(index);
            }
        }
        map.entry(current_sum).or_insert(index + 1);
    }

    match (max_length_start_index, max_length_end_index) {
        (Some(start), Some(end)) => Some(&vec[start..end + 1]),
        _ => None,
    }
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
        assert_eq!(check_if_slice_with_0_exists(&vec![4, 2, -3, 1, 6]), true);
        assert_eq!(check_if_slice_with_0_exists(&vec![4, 2, 0, 1, 6]), true);
        assert_eq!(check_if_slice_with_0_exists(&vec![0]), true);
        assert_eq!(check_if_slice_with_0_exists(&vec![-3, 2, 3, 1, 6]), false);
        assert_eq!(check_if_slice_with_0_exists(&vec![3]), false);
        assert_eq!(check_if_slice_with_0_exists(&vec![]), false);
    }

    #[test]
    fn test_find_max_product_of_2_numbers() {
        assert_eq!(find_max_product_of_2_numbers(&vec![-10, -3, 5, 7, -2]), 35);
        assert_eq!(find_max_product_of_2_numbers(&vec![-10, -3, 5, 4, -2]), 30);
        assert_eq!(find_max_product_of_2_numbers(&vec![-10, -2, -1]), 20);
        assert_eq!(find_max_product_of_2_numbers(&vec![1, 2, 10]), 20);
        assert_eq!(find_max_product_of_2_numbers(&vec![-5, 10]), -50);
        assert_eq!(find_max_product_of_2_numbers(&vec![-5, 0, 10]), 0);
        assert_eq!(find_max_product_of_2_numbers(&vec![-5]), 0);
        assert_eq!(find_max_product_of_2_numbers(&vec![0]), 0);
        assert_eq!(find_max_product_of_2_numbers(&vec![]), 0);
    }

    #[test]
    fn test_find_max_length_slice_with_given_sum() {
        assert_eq!(
            find_max_length_slice_with_given_sum(&vec![5, 6, -5, 5, 3, 4, 1], 7),
            Some(vec![-5, 5, 3, 4].as_slice())
        );

        assert_eq!(
            find_max_length_slice_with_given_sum(&vec![5, 6, -5, 5, 3, 4, 1], 11),
            Some(vec![5, 6, -5, 5].as_slice())
        );

        assert_eq!(
            find_max_length_slice_with_given_sum(&vec![5, 6, -5, 5, 3, 4, 1], 5),
            Some(vec![4, 1].as_slice())
        );

        assert_eq!(
            find_max_length_slice_with_given_sum(&vec![5, 3], 8),
            Some(vec![5, 3].as_slice())
        );

        assert_eq!(
            find_max_length_slice_with_given_sum(&vec![5, 3], 5),
            Some(vec![5].as_slice())
        );

        assert_eq!(
            find_max_length_slice_with_given_sum(&vec![5, 3], 3),
            Some(vec![3].as_slice())
        );

        assert_eq!(
            find_max_length_slice_with_given_sum(&vec![5], 5),
            Some(vec![5].as_slice())
        );

        assert_eq!(
            find_max_length_slice_with_given_sum(&vec![1, 2, 3], 10),
            None
        );

        assert_eq!(find_max_length_slice_with_given_sum(&vec![], 10), None);
    }
}
