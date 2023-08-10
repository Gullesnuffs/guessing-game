use std::{collections::HashMap, time::Duration};

use crate::{
    hashing_strategy::{HashingStrategy, Parameters}, small_strategy::get_small_strategy,
};

pub type BoxIndex = usize;
pub type BoxContent = u8;

pub trait Strategy {
    fn new_encoder_state(&self) -> EncoderState;
    fn n(&self) -> usize;
    fn k(&self) -> u8;
    fn next_snapshot(&self) -> usize;
    fn encode(&self, box_index: BoxIndex, state: &mut EncoderState) -> BoxContent;
    fn decode(&self, box_contents: &[BoxContent]) -> Vec<BoxIndex>;
}

pub struct EncoderState {
    pub num_boxes_seen: usize,
    pub is_old: Vec<bool>,
    pub snapshot: Option<Vec<bool>>,
    pub targets: Option<Vec<BoxContent>>,
    pub coordinate_contraction: HashMap<usize, usize>,
    pub child_state: Option<Box<EncoderState>>,
}

pub fn get_strategy(n: usize) -> Box<dyn Strategy> {
    let s1 = get_small_strategy(1, 1, None);
    let s2 = get_small_strategy(2, 2, Some(Box::new(s1)));
    let s3 = get_small_strategy(5, 3, Some(Box::new(s2)));
    if n == 5 {
        return Box::new(s3);
    }
    let s4 = get_small_strategy(23, 4, Some(Box::new(s3)));
    let parameters = Parameters {
        partial_hashes: 14.0,
        noise_boxes_for_nonzero: -45.0,
        noise_boxes_for_zero: -84.0,
        hash_distribution: vec![0, 100, 100, 100, 0],
        medium_limit: 1100,
        high_limit: 16000,
        time_limit: Duration::from_secs(5),
        min_full_hashes: 20,
        full_hashes_at_start: 5,
    };
    assert!(n > parameters.high_limit);
    let s5 = HashingStrategy::new(
        parameters.medium_limit,
        5,
        Box::new(s4.clone()),
        parameters.clone(),
    );
    let s6 = HashingStrategy::new(parameters.high_limit, 5, Box::new(s5), parameters.clone());
    Box::new(HashingStrategy::new(n, 5, Box::new(s6), parameters.clone()))
}