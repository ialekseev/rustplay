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

// Find a maximum length slice with a given sum
fn find_max_length_slice_with_given_sum(vec: &Vec<i32>, sum: i32) -> Option<&[i32]> {
    let mut map: HashMap<i32, usize> = HashMap::from([(0, 0)]);
    let mut max_length: usize = 0;
    let mut max_length_start_index: Option<usize> = None;
    let mut max_length_end_index: Option<usize> = None;
    let mut current_sum: i32 = 0;

    for (index, elem) in vec.iter().enumerate() {
        current_sum += *elem;

        if let Some(start_position) = map.get(&(current_sum - sum)) {
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

// Sort a binary vector in linear time
fn sort_binary_vector_in_linear_time(vec: &mut Vec<u8>) -> &Vec<u8> {
    let count_0 = vec.iter().fold(0, |acc, e| match e {
        0 => acc + 1,
        1 => acc,
        _ => panic!("The vector is not binary"),
    });

    for i in 0..vec.len() {
        if i < count_0 {
            vec[i] = 0;
        } else {
            vec[i] = 1;
        }
    }

    return vec;
}

fn find_equilibrium_indices_in_vector(vec: &Vec<i32>) -> Vec<usize> {
    let mut map: HashMap<usize, i32> = HashMap::new();

    let mut right_sum: i32 = 0;
    (0..vec.len()).rev().for_each(|index| {
        map.insert(index, right_sum);
        right_sum = right_sum + vec[index];
    });

    let mut left_sum: i32 = 0;
    vec.iter()
        .enumerate()
        .filter(|(index, elem)| {
            let found_equilibrium: bool = left_sum == map[index];
            left_sum = left_sum + *elem;
            found_equilibrium
        })
        .map(|(index, _)| index)
        .collect()
}

// Move all zeros present in a vector to the end
fn move_zeros_to_end(vec: &mut Vec<i32>) -> &Vec<i32> {
    let mut zero_count = 0;

    (0..vec.len()).for_each(|index| {
        if vec[index] == 0 {
            zero_count += 1;
        } else if zero_count > 0 {
            vec[index - zero_count] = vec[index];
            vec[index] = 0;
        }
    });

    return vec;
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

    #[test]
    fn test_sort_binary_vector_in_linear_time() {
        assert_eq!(
            sort_binary_vector_in_linear_time(&mut vec![1, 0, 1, 0, 1, 0, 0, 1]),
            &vec![0, 0, 0, 0, 1, 1, 1, 1]
        );

        assert_eq!(
            sort_binary_vector_in_linear_time(&mut vec![0, 0, 1, 0, 1, 1, 0, 1, 0, 0]),
            &vec![0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
        );

        assert_eq!(
            sort_binary_vector_in_linear_time(&mut vec![0, 0, 1, 1]),
            &vec![0, 0, 1, 1]
        );

        assert_eq!(
            sort_binary_vector_in_linear_time(&mut vec![0, 0, 0]),
            &vec![0, 0, 0]
        );

        assert_eq!(
            sort_binary_vector_in_linear_time(&mut vec![1, 1]),
            &vec![1, 1]
        );

        assert_eq!(sort_binary_vector_in_linear_time(&mut vec![0]), &vec![0]);

        assert_eq!(sort_binary_vector_in_linear_time(&mut vec![1]), &vec![1]);
    }

    #[test]
    fn test_find_equilibrium_indices_in_vector() {
        assert_eq!(
            find_equilibrium_indices_in_vector(&vec![0, -3, 5, -4, -2, 3, 1, 0]),
            vec![0, 3, 7]
        );
        assert_eq!(
            find_equilibrium_indices_in_vector(&vec![2, 3, 5, 1, 2, 2]),
            vec![2]
        );

        assert_eq!(find_equilibrium_indices_in_vector(&vec![3]), vec![0]);

        assert_eq!(find_equilibrium_indices_in_vector(&vec![1, 3, 5]), vec![]);
        assert_eq!(find_equilibrium_indices_in_vector(&vec![1, 2]), vec![])
    }

    #[test]
    fn test_move_zeros_to_end() {
        assert_eq!(
            move_zeros_to_end(&mut vec![5, 0, 0, 2, 3, 0, 4, 0, 1]),
            &vec![5, 2, 3, 4, 1, 0, 0, 0, 0]
        );

        assert_eq!(
            move_zeros_to_end(&mut vec![0, 0, 8, 6, 0, 0]),
            &vec![8, 6, 0, 0, 0, 0]
        );

        assert_eq!(move_zeros_to_end(&mut vec![1, 2, 3]), &vec![1, 2, 3]);
        assert_eq!(move_zeros_to_end(&mut vec![0, 0]), &vec![0, 0]);
        assert_eq!(move_zeros_to_end(&mut vec![1]), &vec![1]);
        assert_eq!(move_zeros_to_end(&mut vec![0]), &vec![0]);
        assert_eq!(move_zeros_to_end(&mut vec![]), &vec![]);
    }
}
