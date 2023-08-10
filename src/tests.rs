use std::time::Instant;

use crate::strategy::{get_strategy, BoxContent, BoxIndex, Strategy};
use rand::{rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};

#[test]
fn test_n_100000() {
    const ITERATIONS: usize = 50000;
    const N: usize = 100_000;
    let strategy = get_strategy(N);
    let mut rng = StdRng::seed_from_u64(2);
    let mut results = Vec::new();
    for _ in 0..ITERATIONS {
        let boxes = get_boxes_to_test(&mut rng, N);
        let last_box_content = rng.gen_range(0..strategy.k());
        let start = Instant::now();
        let result = test_strategy_for_input(strategy.as_ref(), boxes, last_box_content);
        println!("{},{}", result, start.elapsed().as_secs_f64());
        results.push((result, start.elapsed()));
    }
}

fn get_boxes_to_test(rng: &mut StdRng, n: usize) -> Vec<BoxIndex> {
    let mut boxes: Vec<BoxIndex> = vec![];
    for i in 0..n {
        boxes.push(i);
    }
    boxes.shuffle(rng);
    boxes
}

fn test_strategy_for_input(
    strategy: &dyn Strategy,
    boxes: Vec<BoxIndex>,
    last_box_content: BoxContent,
) -> bool {
    let mut contents: Vec<BoxContent> = vec![0; strategy.n()];
    let mut encoder_state = strategy.new_encoder_state();
    for i in 0..strategy.n() {
        if i < strategy.n() - 1 {
            contents[boxes[i]] = strategy.encode(boxes[i], &mut encoder_state);
        }
    }
    contents[boxes[strategy.n() - 1]] = last_box_content;
    let decoded = strategy.decode(&contents);
    assert!(decoded.len() <= 2);
    if !decoded.contains(&boxes[strategy.n() - 1]) {
        return false;
    }
    return true;
}
