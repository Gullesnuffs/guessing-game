use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use sha2::Digest;
use sha2::Sha256;

use crate::injection::get_injection;
use crate::injection::get_injection_inverse;
use crate::strategy::BoxContent;
use crate::strategy::BoxIndex;
use crate::strategy::EncoderState;
use crate::strategy::Strategy;

fn hash_prefixes(bits: &[bool]) -> Vec<Vec<u8>> {
    let mut hasher = Sha256::new();
    let mut hashes: Vec<Vec<u8>> = Vec::new();
    for b in bits {
        hasher.update(if *b { [1] } else { [0] });
        let mut cloned_hasher = hasher.clone();
        let h: Vec<u8> = cloned_hasher.finalize_reset().to_vec();
        hashes.push(h);
    }
    hashes
}

fn reduce_hash_to_box_content(h: &[u8], k: u8, coefficients: &Parameters) -> BoxContent {
    let sum = coefficients
        .hash_distribution
        .iter()
        .fold(0, |acc, x| acc + x);
    let mut s = 0;
    for h in h {
        s = (s * 256 + *h as usize) % sum;
    }
    for i in 0..k {
        if s < coefficients.hash_distribution[i as usize] {
            return i;
        }
        s -= coefficients.hash_distribution[i as usize];
    }
    panic!();
}

pub struct HashingStrategy {
    n: usize,
    k: u8,
    child_strategy: Box<dyn Strategy>,
    random_order: Vec<BoxIndex>,
    parameters: Parameters,
}

impl HashingStrategy {
    fn num_noise_boxes(&self) -> usize {
        self.child_strategy.n() - self.child_strategy.next_snapshot()
    }

    fn num_full_hashes(&self) -> usize {
        self.child_strategy.n().min(
            (((self.child_strategy.n() as f64).sqrt() * 1.0) as usize)
                .max(self.parameters.min_full_hashes),
        )
    }

    fn should_use_full_hash(&self, i: usize) -> bool {
        self.num_full_hashes_seen(i + 1) > self.num_full_hashes_seen(i)
    }

    fn num_full_hashes_seen(&self, i: usize) -> usize {
        let num_full_hashes_at_start = self.num_full_hashes().min(self.parameters.full_hashes_at_start);
        if i < num_full_hashes_at_start {
            return i;
        }
        let fraction_full_hashes_among_remaining = (self.num_full_hashes()
            - num_full_hashes_at_start) as f64
            / (self.child_strategy.n() - num_full_hashes_at_start) as f64;
        num_full_hashes_at_start
            + (((i - num_full_hashes_at_start) as f64) * fraction_full_hashes_among_remaining)
                .ceil() as usize
    }

    pub fn new(
        n: usize,
        k: u8,
        child_strategy: Box<dyn Strategy>,
        priority_coefficients: Parameters,
    ) -> Self {
        let mut random_order: Vec<BoxIndex> = (0..n).collect();
        random_order.shuffle(&mut StdRng::seed_from_u64(47));
        Self {
            n,
            k,
            child_strategy,
            random_order,
            parameters: priority_coefficients,
        }
    }

    fn compute_targets(&self, is_old: &[bool]) -> (Vec<BoxContent>, HashMap<usize, usize>) {
        let is_old_ordered = self
            .random_order
            .iter()
            .map(|&i| is_old[i])
            .collect::<Vec<_>>();
        let hashes = hash_prefixes(&is_old_ordered);
        let mut targets = vec![0; self.n];
        let mut hasher = Sha256::new_with_prefix(
            is_old
                .iter()
                .map(|&b| if b { 1 } else { 0 })
                .collect::<Vec<_>>(),
        );
        let mut num_not_old_seen = 0;
        let mut coordinate_contraction = HashMap::new();
        for i in 0..self.n {
            let j = self.random_order[i];
            if is_old[j] {
                continue;
            }
            coordinate_contraction.insert(j, num_not_old_seen);
            let h = if self.should_use_full_hash(num_not_old_seen) {
                let mut cloned_hasher = hasher.clone();
                let h = cloned_hasher.finalize_reset().to_vec();
                hasher.update(&[1]);
                h
            } else {
                hashes[i].clone()
            };
            targets[j] = reduce_hash_to_box_content(&h, self.k, &self.parameters);
            num_not_old_seen += 1;
        }
        (targets, coordinate_contraction)
    }
}

impl Strategy for HashingStrategy {
    fn new_encoder_state(&self) -> EncoderState {
        EncoderState {
            is_old: vec![false; self.n],
            num_boxes_seen: 0,
            snapshot: None,
            targets: None,
            coordinate_contraction: HashMap::new(),
            child_state: Some(Box::new(self.child_strategy.new_encoder_state())),
        }
    }

    fn n(&self) -> usize {
        self.n
    }

    fn k(&self) -> u8 {
        self.k
    }

    fn next_snapshot(&self) -> usize {
        self.n - self.child_strategy.n()
    }

    fn encode(&self, box_index: BoxIndex, state: &mut EncoderState) -> BoxContent {
        assert!(box_index < self.n());
        state.num_boxes_seen += 1;
        if state.snapshot.is_none() {
            state.is_old[box_index] = true;
        }
        if state.num_boxes_seen == self.next_snapshot() {
            state.snapshot = Some(state.is_old.clone());
            let (targets, coordinate_contraction) = self.compute_targets(&state.is_old);
            state.targets = Some(targets);
            state.coordinate_contraction = coordinate_contraction;
        }
        let result = if state.num_boxes_seen <= self.next_snapshot() {
            0
        } else {
            let injection = get_injection(
                self.child_strategy.k(),
                self.k,
                &[(0, state.targets.as_ref().unwrap()[box_index])],
            );
            let child_box_content = self.child_strategy.encode(
                *state.coordinate_contraction.get(&box_index).unwrap(),
                state.child_state.as_mut().unwrap(),
            );
            injection(child_box_content)
        };
        assert!(result < self.k);
        result
    }

    fn decode(&self, box_contents: &[BoxContent]) -> Vec<BoxIndex> {
        let contents_ordered = self
            .random_order
            .iter()
            .map(|&i| box_contents[i])
            .collect::<Vec<_>>();
        let mut priority_queue = BinaryHeap::new();
        priority_queue.push(Arc::new(DecodingNode::new(self, None, None)));
        let start = Instant::now();
        while let Some(node) = priority_queue.pop() {
            if start.elapsed() > self.parameters.time_limit {
                return vec![];
            }
            if node.num_boxes_seen() == self.n() {
                let mut total_noise = node.num_boxes_seen_of_type[BoxType::NoiseForZero as usize]
                    + node.num_boxes_seen_of_type[BoxType::NoiseForNonZero as usize];
                if total_noise > self.num_noise_boxes() {
                    continue;
                }
                let mut nodes = vec![];
                let mut current_node = node;
                while let Some(parent) = current_node.parent.clone() {
                    nodes.push(current_node.clone());
                    current_node = parent;
                }
                nodes.reverse();
                let mut is_old = vec![false; self.n];
                for (i, node) in nodes.iter().enumerate() {
                    let j = self.random_order[i];
                    if node.box_type.unwrap() == BoxType::Old {
                        is_old[j] = true;
                    }
                }
                let (targets, _) = self.compute_targets(&is_old);
                let mut coordinate_expansion = vec![];
                let mut num_not_old_seen = 0;
                let mut child_box_contents = vec![];
                let mut ok = true;
                let mut undecodable = vec![];
                for (i, node) in nodes.iter().enumerate() {
                    let j = self.random_order[i];
                    if node.box_type.unwrap() != BoxType::Old {
                        let expected_box_content = targets[j];
                        if matches!(node.box_type.unwrap(), BoxType::CurrentWithFullHash)
                            && expected_box_content != contents_ordered[node.num_boxes_seen() - 1]
                        {
                            total_noise += 1;
                            if total_noise > self.num_noise_boxes() {
                                ok = false;
                                break;
                            }
                        }
                        let injection_inverse = get_injection_inverse(
                            self.child_strategy.k(),
                            self.k,
                            &[(0, expected_box_content)],
                        );
                        let child_box_content =
                            injection_inverse(contents_ordered[node.num_boxes_seen() - 1]);
                        if let Some(child_box_content) = child_box_content {
                            child_box_contents.push(child_box_content);
                        } else {
                            undecodable.push((j, child_box_contents.len()));
                            child_box_contents.push(0);
                        }
                        coordinate_expansion.push(j);
                        num_not_old_seen += 1;
                    }
                }
                if undecodable.len() > 1 {
                    ok = false;
                }
                if !ok {
                    continue;
                }
                assert!(self.child_strategy.n() == num_not_old_seen);
                if undecodable.len() > 0 {
                    assert!(undecodable.len() == 1);
                    let (undecodable, index) = undecodable[0];
                    for i in 0..self.child_strategy.k() {
                        let mut child_box_contents = child_box_contents.clone();
                        child_box_contents[index] = i;
                        let decoded = self
                            .child_strategy
                            .decode(&child_box_contents)
                            .into_iter()
                            .map(|i| coordinate_expansion[i])
                            .collect::<Vec<_>>();
                        if !decoded.contains(&undecodable) {
                            ok = false;
                            break;
                        }
                    }
                    if ok {
                        return vec![undecodable];
                    } else {
                        continue;
                    }
                } else {
                    let decoded = self
                        .child_strategy
                        .decode(&child_box_contents)
                        .into_iter()
                        .map(|i| coordinate_expansion[i])
                        .collect::<Vec<_>>();
                    if decoded.is_empty() {
                        continue;
                    }
                    return decoded;
                }
            }
            let mut children = vec![];
            let box_content = contents_ordered[node.num_boxes_seen()];
            if box_content == 0 {
                for box_type in [BoxType::Old, BoxType::NoiseForZero] {
                    let child = DecodingNode::new(self, Some(node.clone()), Some(box_type));
                    children.push(child);
                }
            } else {
                if self.should_use_full_hash(node.num_not_old_boxes_seen()) {
                    children.push(DecodingNode::new(
                        self,
                        Some(node.clone()),
                        Some(BoxType::CurrentWithFullHash),
                    ));
                } else {
                    let noise_child =
                        DecodingNode::new(self, Some(node.clone()), Some(BoxType::NoiseForNonZero));
                    let h = noise_child.hasher.clone().finalize_reset().to_vec();
                    let expected_box_content =
                        reduce_hash_to_box_content(&h, self.k, &self.parameters);
                    if expected_box_content == box_content {
                        children.push(DecodingNode::new(
                            self,
                            Some(node.clone()),
                            Some(BoxType::CurrentWithPartialHash),
                        ));
                    } else {
                        children.push(noise_child);
                    }
                }
            }
            for child in children {
                if child.priority() > f64::NEG_INFINITY {
                    priority_queue.push(Arc::new(child));
                }
            }
        }
        vec![]
    }
}

#[derive(Clone, Copy, Eq, PartialEq, Debug)]
enum BoxType {
    Old,
    CurrentWithFullHash,
    CurrentWithPartialHash,
    NoiseForZero,
    NoiseForNonZero,
}

#[derive(Clone, Debug)]
pub struct Parameters {
    pub partial_hashes: f64,
    pub noise_boxes_for_nonzero: f64,
    pub noise_boxes_for_zero: f64,
    pub hash_distribution: Vec<usize>,
    pub medium_limit: usize,
    pub high_limit: usize,
    pub time_limit: Duration,
    pub min_full_hashes: usize,
    pub full_hashes_at_start: usize,
}

impl Eq for Parameters {}

impl PartialEq for Parameters {
    fn eq(&self, other: &Self) -> bool {
        self.partial_hashes == other.partial_hashes
            && self.noise_boxes_for_nonzero == other.noise_boxes_for_nonzero
            && self.noise_boxes_for_zero == other.noise_boxes_for_zero
            && self.hash_distribution == other.hash_distribution
            && self.medium_limit == other.medium_limit
            && self.high_limit == other.high_limit
            && self.time_limit == other.time_limit
            && self.min_full_hashes == other.min_full_hashes
            && self.full_hashes_at_start == other.full_hashes_at_start
    }
}

struct DecodingNode<'a> {
    strategy: &'a HashingStrategy,
    parent: Option<Arc<DecodingNode<'a>>>,
    box_type: Option<BoxType>,
    num_boxes_seen_of_type: [usize; 5],
    hasher: Sha256,
}

impl<'a> DecodingNode<'a> {
    fn new(
        strategy: &'a HashingStrategy,
        parent: Option<Arc<DecodingNode<'a>>>,
        box_type: Option<BoxType>,
    ) -> Self {
        let (mut hasher, mut num_boxes_seen_of_type) = if let Some(parent) = &parent {
            (parent.hasher.clone(), parent.num_boxes_seen_of_type)
        } else {
            (Sha256::new(), [0; 5])
        };
        if let Some(box_type) = &box_type {
            match box_type {
                BoxType::Old => {
                    hasher.update(&[1]);
                }
                _ => {
                    hasher.update(&[0]);
                }
            }
            num_boxes_seen_of_type[*box_type as usize] += 1;
        }
        Self {
            strategy,
            parent,
            box_type,
            num_boxes_seen_of_type,
            hasher,
        }
    }

    fn num_boxes_seen(&self) -> usize {
        self.num_boxes_seen_of_type.iter().sum()
    }

    fn num_not_old_boxes_seen(&self) -> usize {
        self.num_boxes_seen() - self.num_boxes_seen_of_type[BoxType::Old as usize]
    }

    fn priority(&self) -> f64 {
        let n1 = self.num_boxes_seen();
        let n2 = self.strategy.n() - n1;
        let o1 = self.num_boxes_seen_of_type[BoxType::Old as usize];
        if o1 > self.strategy.next_snapshot() {
            return f64::NEG_INFINITY;
        }
        let o2 = self.strategy.next_snapshot() - o1;
        if o2 > n2 {
            return f64::NEG_INFINITY;
        }

        let not_old = self.num_not_old_boxes_seen();
        if not_old > self.strategy.n() - self.strategy.next_snapshot() {
            return f64::NEG_INFINITY;
        }

        let noise_boxes_for_nonzero =
            self.num_boxes_seen_of_type[BoxType::NoiseForNonZero as usize];
        let noise_boxes_for_zero = self.num_boxes_seen_of_type[BoxType::NoiseForZero as usize];
        if noise_boxes_for_nonzero + noise_boxes_for_zero > self.strategy.num_noise_boxes() {
            return f64::NEG_INFINITY;
        }
        if self.strategy.child_strategy.k() < self.strategy.k() && noise_boxes_for_zero > 1 {
            return f64::NEG_INFINITY;
        }
        let partial_hashes = self.num_boxes_seen_of_type[BoxType::CurrentWithPartialHash as usize];
        (partial_hashes as f64) * self.strategy.parameters.partial_hashes
            + (noise_boxes_for_nonzero as f64) * self.strategy.parameters.noise_boxes_for_nonzero
            + (noise_boxes_for_zero as f64) * self.strategy.parameters.noise_boxes_for_zero
    }
}

impl Ord for DecodingNode<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority().partial_cmp(&other.priority()).unwrap()
    }
}

impl PartialOrd for DecodingNode<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.priority().partial_cmp(&other.priority())
    }
}

impl Eq for DecodingNode<'_> {}

impl PartialEq for DecodingNode<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.priority() == other.priority()
    }
}
