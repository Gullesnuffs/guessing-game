use itertools::Itertools;

use crate::strategy::get_strategy;

mod hashing_strategy;
mod injection;
mod small_strategy;
mod strategy;
#[cfg(test)]
mod tests;
mod util;

fn main() {
    const N: usize = 100_000;
    let strategy = get_strategy(N);
    let mut input = String::new();
    let stdin = std::io::stdin();
    stdin.read_line(&mut input).unwrap();
    let split = input.split(" ").collect_vec();
    let p: u8 = split[0].trim().parse().unwrap();
    let n: usize = split[1].trim().parse().unwrap();
    assert!(n <= N);
    let mut encoder_state = strategy.new_encoder_state();
    let mut preencoded_boxes = vec![];
    for i in n..N {
        preencoded_boxes.push(strategy.encode(i, &mut encoder_state));
    }
    if p == 1 {
        println!("{}", strategy.k());
        for _ in 0..(n - 1) {
            let index: usize;
            input.clear();
            stdin.read_line(&mut input).unwrap();
            index = input.trim().parse().unwrap();
            println!("{}", strategy.encode(index, &mut encoder_state));
        }
    } else {
        input.clear();
        stdin.read_line(&mut input).unwrap();
        let mut contents = input
            .trim()
            .split_whitespace()
            .map(|x| x.parse().unwrap())
            .collect_vec();
        contents.extend(preencoded_boxes);
        assert_eq!(contents.len(), N);
        let decoded = strategy.decode(&contents).into_iter().filter(|x| *x < n);
        println!("{}", decoded.into_iter().join(" "));
    }
}
