/// Returns an injective function from a set of size `from_set_size` to a set of size `to_set_size`, with the values of `predefined_values` fixed.
pub fn get_injection(from_set_size: u8, to_set_size: u8, predefined_values: &[(u8, u8)]) -> impl Fn(u8) -> u8 {
    assert!(from_set_size <= to_set_size);
    let mut values = vec![to_set_size; from_set_size as usize];
    for (from, to) in predefined_values {
        values[*from as usize] = *to;
    }
    for i in 0..(from_set_size as usize) {
        if values[i] == to_set_size {
            let mut j = to_set_size - 1;
            while values.contains(&j) {
                j -= 1;
            }
            values[i] = j;
        }
    }
    for (from, to) in predefined_values {
        assert_eq!(values[*from as usize], *to);
    }
    for i in 0..(from_set_size as usize) {
        assert!(values[i] < to_set_size);
    }
    let value_set = values.iter().collect::<std::collections::HashSet<_>>();
    assert_eq!(value_set.len(), from_set_size as usize);
    move |x| values[x as usize]
}

pub fn get_injection_inverse(from_set_size: u8, to_set_size: u8, predefined_values: &[(u8, u8)]) -> impl Fn(u8) -> Option<u8> {
    let injection = get_injection(from_set_size, to_set_size, predefined_values);
    let mut values = vec![None; to_set_size as usize];
    for i in 0..from_set_size {
        values[injection(i) as usize] = Some(i);
    }
    move |x| values[x as usize]
}