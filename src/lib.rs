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

// Find equilibrium indices in a vector
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

// Find majority element (Boyer–Moore Majority Vote Algorithm)
fn find_majority_element(vec: &Vec<i32>) -> Option<i32> {
    let mut m = 0;
    let mut i = 0;
    //first pass:
    vec.iter().for_each(|&x| {
        if i == 0 {
            m = x;
            i = 1;
        } else if m == x {
            i = i + 1;
        } else {
            i = i - 1;
        }
    });
    //second pass:
    let count = vec
        .iter()
        .fold(0, |count, &x| if x == m { count + 1 } else { count });
    //return Some(m) if actual majority, None otherwise:
    (count > vec.len() / 2).then(|| m)
}

// Find maximum sum slice
fn find_max_sum_slice(vec: &Vec<i32>) -> &[i32] {
    if vec.is_empty() {
        return &[];
    }

    let mut best_sum = vec[0];
    let mut best_start_index = 0;
    let mut best_end_index = 0;

    let mut current_sum = vec[0];
    let mut current_start_index = 0;

    (1..vec.len()).for_each(|i| {
        if current_sum <= 0 {
            //if the current sum is <= 0, we start tracking a new sequence from the current element
            current_start_index = i;
            current_sum = vec[i];
        } else {
            //otherwise we continue the currently tracked sequence
            current_sum += vec[i];
        }
        //then we update the best sum if the current sum is greater
        if current_sum >= best_sum {
            best_sum = current_sum;
            best_start_index = current_start_index;
            best_end_index = i;
        }
    });

    &vec[best_start_index..best_end_index + 1]
}

// find pairs with provided difference in a vector
fn find_pairs_with_difference(vec: &Vec<i32>, diff: i32) -> Vec<(i32, i32)> {
    let mut set = HashSet::new();

    vec.iter().for_each(|e| {
        set.insert(e);
    });

    vec.iter()
        .flat_map(|&e| {
            let another = e + diff;
            if set.contains(&another) {
                set.remove(&another);
                Some((e, another))
            } else {
                None
            }
        })
        .collect()
}

// partition a vector into 2 slices with equal sum
fn partition_vector_into_2_slices_with_equal_sum(vec: &Vec<i32>) -> Option<(&[i32], &[i32])> {
    let sum_all = vec.iter().fold(0, |sum, x| sum + x);

    let mut sum_left = 0;
    (0..vec.len()).find_map(|i| {
        let sum_right = sum_all - sum_left;
        let found = if sum_left == sum_right {
            Some((&vec[0..i], &vec[i..vec.len()]))
        } else {
            None
        };
        sum_left += vec[i];
        found
    })
}

// quicksort implementation
fn quick_sort(vec: &mut Vec<i32>) -> &Vec<i32> {
    fn partition(slice: &mut [i32]) -> usize {
        let pivot_idx = slice.len() - 1;
        let pivot = slice[pivot_idx];
        let mut next_swap_idx = 0;

        (0..slice.len()).for_each(|idx| {
            if slice[idx] < pivot {
                slice.swap(next_swap_idx, idx);
                next_swap_idx += 1;
            }
        });
        slice.swap(pivot_idx, next_swap_idx);

        next_swap_idx
    }

    fn qsort(slice: &mut [i32]) {
        let pivot_idx = partition(slice);
        let len = slice.len();
        if pivot_idx > 0 {
            qsort(&mut slice[0..pivot_idx]);
        }
        if pivot_idx < len - 1 {
            qsort(&mut slice[pivot_idx + 1..len]);
        }
    }

    if !vec.is_empty() {
        qsort(&mut vec[..]);
    }

    vec
}

// Find the minimum index of a repeating element in a vector
fn find_min_index_of_repeating_element_in_vector(vec: &Vec<i32>) -> Option<usize> {
    let mut set = HashSet::new();

    (0..vec.len())
        .rev()
        .flat_map(|i| {
            let found = if set.contains(&vec[i]) { Some(i) } else { None };
            set.insert(vec[i]);
            found
        })
        .last()
}

// Find a pivot index in a vector (before which all elements are smaller and after which all are greater)
fn find_pivot_index_in_vector(vec: &Vec<i32>) -> Option<usize> {
    let mut map = HashMap::new();

    //first (reversed) traversal: for each index save to the map current min value
    (0..vec.len()).rev().fold(i32::MAX, |min, i| {
        map.insert(i, min);
        if vec[i] < min {
            vec[i]
        } else {
            min
        }
    });

    //second traversal: find the pivot index
    let mut max_left = i32::MIN;
    (0..vec.len()).find_map(|i| {
        let min_right = map[&i];
        let found = if vec[i] > max_left && vec[i] < min_right {
            Some(i)
        } else {
            None
        };
        if vec[i] > max_left {
            max_left = vec[i]
        }
        found
    })
}

// Check if a number is a palindrome (no convertion to string is allowed)
fn is_palindrome(mut num: i32) -> bool {
    match num {
        n if n == 0 => return true,       //0 is palindrome
        n if n < 0 => return false,       //negative num is not
        n if n % 10 == 0 => return false, //num ending 0 is not
        _ => (),
    }

    //reversing the half of the num (reversing the whole number could
    //cause int overflow):
    let mut num_reversed = 0;
    while num > num_reversed {
        num_reversed = num_reversed * 10 + num % 10;
        num /= 10;
    }

    //if num originally had an even number of digits e.g. 1221 then
    //[num == num_reversed == 12].
    //if num had an odd number of digits e.g. 12321 then
    //[num == 12, num_reversed = 123], the mid number (3)
    //could be ignored in this case.
    num == num_reversed || num == num_reversed / 10
}

// Reverse a number (no conversion to string is allowed)
fn reverse_number(mut num: i32) -> i32 {
    let mut num_reversed: i32 = 0;
    while num != 0 {
        if num_reversed.abs() > i32::MAX / 10 {
            return 0;
        } else {
            num_reversed = num_reversed * 10 + num % 10;
            num /= 10;
        }
    }
    num_reversed
}

// Find the longest common prefix string
fn find_longest_common_prefix_string<'a>(vec: &Vec<&'a str>) -> &'a str {
    fn common_prefix<'a>(s1: &'a str, s2: &'a str) -> &'a str {
        let prefix_end_index = s1
            .chars()
            .zip(s2.chars())
            .enumerate()
            .find(|(_, (char1, char2))| char1 != char2)
            .map(|(index, _)| index)
            .unwrap_or(usize::min(s1.len(), s2.len()));
        &s1[0..prefix_end_index]
    }

    if vec.is_empty() {
        return "";
    }

    (1..vec.len()).fold(vec[0], |prx, i| common_prefix(prx, vec[i]))
}

// Given a string s containing just the characters '{', '}', '[', ']', '(', ')' determine if the input string has correct brackets.
fn has_string_valid_brackets(str: &str) -> bool {
    const OPEN_BRACKETS: &str = "{[(";
    const CLOSE_BRACKETS: &str = "}])";
    let mut stack = Vec::new();

    for char in str.chars() {
        if OPEN_BRACKETS.contains(|c| c == char) {
            stack.push(char)
        } else if CLOSE_BRACKETS.contains(|c| c == char) {
            let popped = stack.pop();
            match (popped, char) {
                (Some('('), ')') | (Some('{'), '}') | (Some('['), ']') => continue,
                _ => return false,
            }
        } else {
            panic!("the input string must only contain brackets!")
        }
    }

    return stack.is_empty();
}

// Given a non-negative integer num, return the square root of num rounded down to the nearest integer.
fn sqrt(num: i32) -> i32 {
    assert!(num >= 0);

    let mut left = 0;
    let mut right = num;
    while left < right {
        let mid = (left + right + 1) / 2;
        if mid * mid > num {
            right = mid - 1;
        } else {
            left = mid;
        }
    }
    right
}

// Check whether provided strings are anagrams or not
fn check_if_anagram_strings(s1: &str, s2: &str) -> bool {
    let mut map1 = HashMap::new();
    let mut map2 = HashMap::new();

    s1.chars()
        .filter(|&c| c != ' ')
        .zip(s2.chars().filter(|&c| c != ' '))
        .for_each(|(char1, char2)| {
            let count1 = map1.entry(char1.to_ascii_lowercase()).or_insert(0);
            *count1 += 1;
            let count2 = map2.entry(char2.to_ascii_lowercase()).or_insert(0);
            *count2 += 1;
        });

    map1 == map2
}

// Given a vector of n-1 distinct integers in the range of 1 to n, find the missing number in it in linear time.
fn find_missing_number_in_vector(vec: &Vec<i32>) -> i32 {
    let sum_vec: i32 = vec.iter().sum();

    let n = (vec.len() + 1) as i32;
    let sum_nat = n * (n + 1) / 2;

    sum_nat - sum_vec
}

// Given two integers, where n is non-negative, raise a value b to the power of n.
fn pow(mut b: i32, mut n: u32) -> i32 {
    let mut pow: i32 = 1;

    while n > 0 {
        if (n % 2) == 1 {
            pow *= b;
        }

        n /= 2;
        b *= b;
    }

    pow
}

// Calculate nth Fibonacci number.
fn fibonacci_number(n: u32) -> u32 {
    if n <= 1 {
        return n;
    }

    (2..=n)
        .fold((0, 1), |(prev, curr), _| (curr, prev + curr))
        .1
}

// Remove adjacent duplicate characters from a string
fn remove_adjacent_duplicate_chars(str: &str) -> String {
    let mut prev_char: Option<char> = None;

    str.chars()
        .filter(|&char| {
            if Some(char) != prev_char {
                prev_char = Some(char);
                true
            } else {
                false
            }
        })
        .collect()
}

// Check if a provided number is a prime number or not
fn is_prime(n: u32) -> bool {
    match n {
        _ if n <= 1 => return false,     // prime num has to be > 1
        _ if n == 2 => return true,      // 2 is prime num
        _ if n % 2 == 0 => return false, //other even nums could not be prime
        _ => (),
    }

    let sqrt_n = (n as f32).sqrt() as u32;

    // iterate over odd numbers i=3,5,7... while i <= sqrt(n)
    // if n is divisible by some i, then n is not prime num
    let mut i = 3;
    while i <= sqrt_n {
        if n % i == 0 {
            return false;
        }
        i += 2;
    }

    true
}

// Given a non-negative integer n, return true if it is a power of two. Otherwise, return false.
fn power_of_2(n: u32) -> bool {
    (n != 0) && (n & n - 1 == 0)
}

// Given two strings, determine whether they are isomorphic.
fn is_isomorphic_strings(s1: &str, s2: &str) -> bool {
    if s1.len() != s2.len() {
        return false;
    }

    let mut map1 = HashMap::new();
    let mut map2 = HashMap::new();
    s1.chars()
        .zip(s2.chars())
        .all(|(c1, c2)| match (map1.get(&c1), map2.get(&c2)) {
            (Some(&c2_from_map1), _) if c2_from_map1 != c2 => false,
            (_, Some(&c1_from_map2)) if c1_from_map2 != c1 => false,
            _ => {
                map1.insert(c1, c2);
                map2.insert(c2, c1);
                true
            }
        })
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
    fn test_check_if_slice_with_0_exists() {
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

    #[test]
    fn test_find_majority_element() {
        assert_eq!(
            find_majority_element(&vec![4, 8, 7, 4, 4, 5, 4, 3, 1, 4, 4]),
            Some(4)
        );
        assert_eq!(find_majority_element(&vec![1, 3, 3]), Some(3));
        assert_eq!(find_majority_element(&vec![1, 3, 3, 3]), Some(3));
        assert_eq!(find_majority_element(&vec![1]), Some(1));
        assert_eq!(
            find_majority_element(&vec![4, 8, 7, 4, 4, 5, 4, 3, 1, 4]),
            None
        );
        assert_eq!(find_majority_element(&vec![1, 2]), None);
        assert_eq!(find_majority_element(&vec![]), None);
    }

    #[test]
    fn test_find_max_sum_slice() {
        assert_eq!(
            find_max_sum_slice(&vec![-2, 1, -3, 4, -1, 2, 1, -5, 4]),
            vec![4, -1, 2, 1].as_slice()
        );

        assert_eq!(
            find_max_sum_slice(&vec![-4, 1, 2, 3, -5]),
            vec![1, 2, 3].as_slice()
        );

        assert_eq!(
            find_max_sum_slice(&vec![1, 2, 3, -5]),
            vec![1, 2, 3].as_slice()
        );

        assert_eq!(
            find_max_sum_slice(&vec![-5, 1, 2, 3]),
            vec![1, 2, 3].as_slice()
        );

        assert_eq!(find_max_sum_slice(&vec![-4, 1, -5]), vec![1].as_slice());

        assert_eq!(find_max_sum_slice(&vec![-4, 0, -5]), vec![0].as_slice());

        assert_eq!(find_max_sum_slice(&vec![1, 2, 3]), vec![1, 2, 3].as_slice());

        assert_eq!(find_max_sum_slice(&vec![-4, -3, -5]), vec![-3].as_slice());

        assert_eq!(find_max_sum_slice(&vec![1]), vec![1].as_slice());
        assert_eq!(find_max_sum_slice(&vec![-1]), vec![-1].as_slice());
        assert_eq!(find_max_sum_slice(&vec![]), vec![].as_slice());
    }

    #[test]
    fn test_find_pairs_with_difference() {
        assert_eq!(
            find_pairs_with_difference(&vec![1, 5, 2, 2, 2, 5, 5, 4], 3),
            vec![(1, 4), (2, 5)]
        );

        assert_eq!(find_pairs_with_difference(&vec![1, 5], 4), vec![(1, 5)]);
        assert_eq!(find_pairs_with_difference(&vec![1, 5, 6], 2), vec![]);
        assert_eq!(find_pairs_with_difference(&vec![1], 3), vec![]);
        assert_eq!(find_pairs_with_difference(&vec![0], 3), vec![]);
    }

    #[test]
    fn test_partition_vector_into_2_slices_with_equal_sum() {
        assert_eq!(
            partition_vector_into_2_slices_with_equal_sum(&vec![7, -5, -4, 2, 4]),
            Some((vec![7, -5].as_slice(), vec![-4, 2, 4].as_slice()))
        );

        assert_eq!(
            partition_vector_into_2_slices_with_equal_sum(&vec![7, -6, 3, -5, 1]),
            Some((vec![].as_slice(), vec![7, -6, 3, -5, 1].as_slice()))
        );

        assert_eq!(
            partition_vector_into_2_slices_with_equal_sum(&vec![2, 2]),
            Some((vec![2].as_slice(), vec![2].as_slice()))
        );

        assert_eq!(
            partition_vector_into_2_slices_with_equal_sum(&vec![0]),
            Some((vec![].as_slice(), vec![0].as_slice()))
        );

        assert_eq!(
            partition_vector_into_2_slices_with_equal_sum(&vec![1, 3, 5, 7]),
            None
        );

        assert_eq!(
            partition_vector_into_2_slices_with_equal_sum(&vec![2]),
            None
        );

        assert_eq!(partition_vector_into_2_slices_with_equal_sum(&vec![]), None);
    }

    #[test]
    fn test_quick_sort() {
        assert_eq!(
            quick_sort(&mut vec![-4, 1, 25, 50, 8, 10, 23]),
            &vec![-4, 1, 8, 10, 23, 25, 50]
        );

        assert_eq!(
            quick_sort(&mut vec![1, 0, 9, 8, 10, 30, 5, 6, 7, 4, 3, 2, 1]),
            &vec![0, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 30]
        );

        assert_eq!(
            quick_sort(&mut vec![2, 1, 6, 10, 4, 1, 3, 9, 7]),
            &vec![1, 1, 2, 3, 4, 6, 7, 9, 10]
        );

        assert_eq!(quick_sort(&mut vec![4, 1, 3, 9, 7]), &vec![1, 3, 4, 7, 9]);

        assert_eq!(quick_sort(&mut vec![4, 2, 6, 1]), &vec![1, 2, 4, 6]);

        assert_eq!(quick_sort(&mut vec![1, 2, 3]), &vec![1, 2, 3]);

        assert_eq!(quick_sort(&mut vec![1, 2]), &vec![1, 2]);

        assert_eq!(quick_sort(&mut vec![1]), &vec![1]);

        assert_eq!(quick_sort(&mut vec![]), &vec![]);
    }

    #[test]
    fn test_find_min_index_of_repeating_element_in_vector() {
        assert_eq!(
            find_min_index_of_repeating_element_in_vector(&vec![6, 7, 4, 5, 4, 7, 5]),
            Some(1)
        );
        assert_eq!(
            find_min_index_of_repeating_element_in_vector(&vec![1, 2, 5, 3, 4, 7, 3, 5, 8, 9]),
            Some(2)
        );
        assert_eq!(
            find_min_index_of_repeating_element_in_vector(&vec![1, 1]),
            Some(0)
        );
        assert_eq!(
            find_min_index_of_repeating_element_in_vector(&vec![1, 2, 3, 4, 5, 6]),
            None
        );
        assert_eq!(
            find_min_index_of_repeating_element_in_vector(&vec![1]),
            None
        );
        assert_eq!(find_min_index_of_repeating_element_in_vector(&vec![]), None);
    }

    #[test]
    fn test_find_pivot_index_in_vector() {
        assert_eq!(
            find_pivot_index_in_vector(&vec![5, 3, 4, 6, 2, 7, 10, 8]),
            Some(5)
        );
        assert_eq!(
            find_pivot_index_in_vector(&vec![3, 1, 12, 10, 23, 50, 25]),
            Some(4)
        );
        assert_eq!(
            find_pivot_index_in_vector(&vec![-4, 1, 25, 50, 8, 10, 23]),
            Some(0)
        );
        assert_eq!(find_pivot_index_in_vector(&vec![3, 1, 8, 10]), Some(2));
        assert_eq!(find_pivot_index_in_vector(&vec![1, 2, 3]), Some(0));
        assert_eq!(find_pivot_index_in_vector(&vec![1]), Some(0));
        assert_eq!(find_pivot_index_in_vector(&vec![3, 2, 1]), None);
        assert_eq!(find_pivot_index_in_vector(&vec![]), None);
    }

    #[test]
    fn test_is_palindrome() {
        assert_eq!(is_palindrome(121), true);
        assert_eq!(is_palindrome(1), true);
        assert_eq!(is_palindrome(11), true);
        assert_eq!(is_palindrome(111), true);
        assert_eq!(is_palindrome(212), true);
        assert_eq!(is_palindrome(22122), true);
        assert_eq!(is_palindrome(112211), true);
        assert_eq!(is_palindrome(0), true);

        assert_eq!(is_palindrome(12), false);
        assert_eq!(is_palindrome(123), false);
        assert_eq!(is_palindrome(10), false);
        assert_eq!(is_palindrome(-1), false);
        assert_eq!(is_palindrome(-121), false);
        assert_eq!(is_palindrome(i32::MAX), false);
    }

    #[test]
    fn test_reverse_number() {
        assert_eq!(reverse_number(1234), 4321);
        assert_eq!(reverse_number(120), 21);
        assert_eq!(reverse_number(1), 1);
        assert_eq!(reverse_number(0), 0);
        assert_eq!(reverse_number(-1), -1);
        assert_eq!(reverse_number(-1234), -4321);
        assert_eq!(reverse_number(123456789), 987654321);
        assert_eq!(reverse_number(1463847412), 2147483641);
        assert_eq!(reverse_number(1463847413), 0);
        assert_eq!(reverse_number(-1463847413), 0);
        assert_eq!(reverse_number(i32::MAX), 0);
        assert_eq!(reverse_number(i32::MIN), 0);
    }

    #[test]
    fn test_find_longest_common_prefix_string() {
        assert_eq!(
            find_longest_common_prefix_string(&vec!["abcd", "abc"]),
            "abc"
        );
        assert_eq!(
            find_longest_common_prefix_string(&vec!["abcd", "abcdef", "abc", "abcde"]),
            "abc"
        );
        assert_eq!(find_longest_common_prefix_string(&vec!["def"]), "def");
        assert_eq!(
            find_longest_common_prefix_string(&vec!["abcd", "abcd", "abcd"]),
            "abcd"
        );
        assert_eq!(
            find_longest_common_prefix_string(&vec!["a", "abc", "ab"]),
            "a"
        );
        assert_eq!(
            find_longest_common_prefix_string(&vec!["abc", "defg", "hij"]),
            ""
        );
        assert_eq!(find_longest_common_prefix_string(&vec![]), "");
    }

    #[test]
    fn test_has_string_valid_brackets() {
        assert_eq!(has_string_valid_brackets("[]{}()"), true);
        assert_eq!(has_string_valid_brackets("{([[]])}"), true);
        assert_eq!(has_string_valid_brackets("[]"), true);
        assert_eq!(has_string_valid_brackets(""), true);
        assert_eq!(has_string_valid_brackets("{([)}"), false);
        assert_eq!(has_string_valid_brackets("{]"), false);
        assert_eq!(has_string_valid_brackets("}"), false);
        assert_eq!(has_string_valid_brackets("{"), false);
        assert_eq!(has_string_valid_brackets("}}"), false);
        assert_eq!(has_string_valid_brackets("{{"), false);
    }

    #[test]
    fn test_sqrt() {
        assert_eq!(sqrt(0), 0);
        assert_eq!(sqrt(1), 1);
        assert_eq!(sqrt(2), 1);
        assert_eq!(sqrt(3), 1);
        assert_eq!(sqrt(4), 2);
        assert_eq!(sqrt(5), 2);
        assert_eq!(sqrt(6), 2);
        assert_eq!(sqrt(7), 2);
        assert_eq!(sqrt(8), 2);
        assert_eq!(sqrt(9), 3);
        assert_eq!(sqrt(16), 4);
        assert_eq!(sqrt(17), 4);
        assert_eq!(sqrt(225), 15);
    }

    #[test]
    fn test_check_if_anagram_strings() {
        assert_eq!(
            check_if_anagram_strings("New York Times", "monkeys write"),
            true
        );

        assert_eq!(
            check_if_anagram_strings("McDonald's restaurants", "Uncle Sam's standard rot"),
            true
        );

        assert_eq!(check_if_anagram_strings("coronavirus", "carnivorous"), true);

        assert_eq!(check_if_anagram_strings("daddy", "daddy"), true);
        assert_eq!(check_if_anagram_strings("a", "a"), true);
        assert_eq!(check_if_anagram_strings("", ""), true);

        assert_eq!(check_if_anagram_strings("daddy", "mummy"), false)
    }

    #[test]
    fn test_find_missing_number_in_vector() {
        assert_eq!(find_missing_number_in_vector(&vec![1, 2, 3, 5, 6, 7]), 4);
        assert_eq!(find_missing_number_in_vector(&vec![2, 3]), 1);
        assert_eq!(find_missing_number_in_vector(&vec![1, 3]), 2);
        assert_eq!(find_missing_number_in_vector(&vec![2]), 1);
    }

    #[test]
    fn test_pow() {
        assert_eq!(pow(2, 4), 16);
        assert_eq!(pow(-2, 3), -8);
        assert_eq!(pow(6, 3), 216);
        assert_eq!(pow(-2, 10), 1024);
        assert_eq!(pow(-3, 4), 81);
        assert_eq!(pow(5, 0), 1);
    }

    #[test]
    fn test_fibonacci_number() {
        assert_eq!(fibonacci_number(0), 0);
        assert_eq!(fibonacci_number(1), 1);
        assert_eq!(fibonacci_number(2), 1);
        assert_eq!(fibonacci_number(3), 2);
        assert_eq!(fibonacci_number(4), 3);
        assert_eq!(fibonacci_number(5), 5);
        assert_eq!(fibonacci_number(6), 8);
        assert_eq!(fibonacci_number(7), 13);
        assert_eq!(fibonacci_number(8), 21);
        assert_eq!(fibonacci_number(9), 34);
        assert_eq!(fibonacci_number(10), 55);
    }

    #[test]
    fn test_remove_adjacent_duplicate_chars() {
        assert_eq!(remove_adjacent_duplicate_chars("aaabccccddde"), "abcde");
        assert_eq!(remove_adjacent_duplicate_chars("abbbc"), "abc");
        assert_eq!(remove_adjacent_duplicate_chars("aaa"), "a");
        assert_eq!(remove_adjacent_duplicate_chars("a"), "a");
        assert_eq!(remove_adjacent_duplicate_chars(""), "");
    }

    #[test]
    fn test_is_prime() {
        assert_eq!(is_prime(0), false);
        assert_eq!(is_prime(1), false);
        assert_eq!(is_prime(2), true);
        assert_eq!(is_prime(3), true);
        assert_eq!(is_prime(4), false);
        assert_eq!(is_prime(5), true);
        assert_eq!(is_prime(6), false);
        assert_eq!(is_prime(7), true);
        assert_eq!(is_prime(8), false);
        assert_eq!(is_prime(9), false);
        assert_eq!(is_prime(10), false);
        assert_eq!(is_prime(11), true);
        assert_eq!(is_prime(12), false);
        assert_eq!(is_prime(13), true);
    }

    #[test]
    fn test_power_of_2() {
        assert_eq!(power_of_2(0), false);
        assert_eq!(power_of_2(1), true);
        assert_eq!(power_of_2(2), true);
        assert_eq!(power_of_2(3), false);
        assert_eq!(power_of_2(4), true);
        assert_eq!(power_of_2(5), false);
        assert_eq!(power_of_2(6), false);
        assert_eq!(power_of_2(7), false);
        assert_eq!(power_of_2(8), true);
    }

    #[test]
    fn test_is_isomorphic_strings() {            
        assert_eq!(is_isomorphic_strings("abcdae", "zbcdzy"), true);
        assert_eq!(is_isomorphic_strings("cat", "dog"), true);
        assert_eq!(is_isomorphic_strings("abc", "def"), true);
        assert_eq!(is_isomorphic_strings("a", "b"), true);
        assert_eq!(is_isomorphic_strings("aaa", "bbb"), true);
        assert_eq!(is_isomorphic_strings("aab", "aac"), true);

        assert_eq!(is_isomorphic_strings("madc", "mama"), false);
        assert_eq!(is_isomorphic_strings("mama", "madc"), false);
        assert_eq!(is_isomorphic_strings("aac", "abb"), false);
        assert_eq!(is_isomorphic_strings("abc", "abcd"), false);
        assert_eq!(is_isomorphic_strings("abcd", "abc"), false);
    }
}
