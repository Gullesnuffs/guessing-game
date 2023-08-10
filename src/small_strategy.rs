use std::{
    collections::{HashMap, HashSet},
    fs::File,
};

use queues::{IsQueue, Queue};
use rand::{rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};
use serde::{Deserialize, Serialize};

use crate::{
    injection::{get_injection, get_injection_inverse},
    strategy::{BoxContent, BoxIndex, EncoderState, Strategy},
    util::subsets_of_size,
};

#[derive(Hash, PartialEq, Eq, Clone, Debug)]
pub struct SmallStrategyOutput {
    pub is_old: Vec<bool>,
    pub box_contents: Vec<BoxContent>,
    pub last_box: BoxIndex,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct SmallStrategy {
    n: usize,
    k: u8,
    child_strategy: Option<Box<SmallStrategy>>,
    pub target_map: HashMap<Vec<bool>, Vec<BoxContent>>,
}

impl SmallStrategy {
    pub fn new_random(n: usize, k: u8, child_strategy: Option<Box<SmallStrategy>>) -> Self {
        let subsets = subsets_of_size(n, n - child_strategy.as_ref().map_or(0, |s| s.n()));
        Self {
            n,
            k,
            child_strategy: child_strategy.clone(),
            target_map: subsets
                .clone()
                .into_iter()
                .map(|is_old| {
                    let mut rng = StdRng::seed_from_u64(42);
                    let range = if k == 1 { 0..1 } else { 1..k };
                    let mut targets: Vec<BoxContent> =
                        (0..n).map(|_| rng.gen_range(range.clone())).collect();
                    for i in 0..n {
                        if is_old[i] {
                            targets[i] = 0;
                        }
                    }
                    (is_old, targets)
                })
                .collect(),
        }
    }

    fn compute_targets(&self, is_old: &[bool]) -> (Vec<BoxContent>, HashMap<usize, usize>) {
        let mut num_not_old_seen = 0;
        let mut coordinate_contraction = HashMap::new();
        for i in 0..self.n {
            if !is_old[i] {
                coordinate_contraction.insert(i, num_not_old_seen);
                num_not_old_seen += 1;
            }
        }
        let targets = self.target_map.get(is_old).unwrap().to_vec();
        (targets, coordinate_contraction)
    }

    pub fn get_all_outputs(&self) -> HashSet<SmallStrategyOutput> {
        if self.n > 1 {
            let mut outputs = HashSet::new();
            let child_strategy = self.child_strategy.as_ref().unwrap();
            for is_old in self.target_map.keys() {
                let target = self.target_map.get(is_old).unwrap();
                for output in &child_strategy.get_all_outputs() {
                    let mut expanded_box_contents = vec![0; self.n];
                    let mut num_not_old_seen = 0;
                    let mut last_box = None;
                    for i in 0..self.n {
                        if is_old[i] {
                            expanded_box_contents[i] = 0;
                        } else {
                            let injection =
                                get_injection(child_strategy.k(), self.k, &[(0, target[i])]);
                            expanded_box_contents[i] =
                                injection(output.box_contents[num_not_old_seen]);
                            if num_not_old_seen == output.last_box {
                                last_box = Some(i);
                            }
                            num_not_old_seen += 1;
                        }
                    }
                    for last_box_value in 0..self.k {
                        expanded_box_contents[last_box.unwrap()] = last_box_value;
                        outputs.insert(SmallStrategyOutput {
                            is_old: is_old.to_owned(),
                            box_contents: expanded_box_contents.clone(),
                            last_box: last_box.unwrap(),
                        });
                    }
                }
            }
            outputs
        } else {
            assert!(self.n == 1);
            let mut outputs = HashSet::new();
            for i in 0..self.k {
                outputs.insert(SmallStrategyOutput {
                    is_old: vec![false],
                    box_contents: vec![i],
                    last_box: 0,
                });
            }
            outputs
        }
    }
}

impl Strategy for SmallStrategy {
    fn new_encoder_state(&self) -> EncoderState {
        EncoderState {
            is_old: vec![false; self.n],
            num_boxes_seen: 0,
            snapshot: None,
            targets: None,
            coordinate_contraction: HashMap::new(),
            child_state: self
                .child_strategy
                .as_ref()
                .map(|s| Box::new(s.new_encoder_state())),
        }
    }

    fn n(&self) -> usize {
        self.n
    }

    fn k(&self) -> u8 {
        self.k
    }

    fn next_snapshot(&self) -> usize {
        if let Some(child_strategy) = &self.child_strategy {
            self.n - child_strategy.n()
        } else {
            self.n
        }
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
            let child_strategy = self.child_strategy.as_ref().unwrap();
            let injection = get_injection(
                child_strategy.k(),
                self.k,
                &[(0, state.targets.as_ref().unwrap()[box_index])],
            );
            let child_box_content = child_strategy.encode(
                *state.coordinate_contraction.get(&box_index).unwrap(),
                state.child_state.as_mut().unwrap(),
            );
            injection(child_box_content)
        };
        result
    }

    fn decode(&self, box_contents: &[BoxContent]) -> Vec<BoxIndex> {
        if self.child_strategy.is_none() {
            return (0..self.n).collect();
        }
        let mut options = HashSet::new();
        let child_strategy = self.child_strategy.as_ref().unwrap();
        for (candidate_is_old, candidate_targets) in &self.target_map {
            let mut ok = true;
            let mut child_box_contents = vec![];
            let mut coordinate_expansion = vec![];
            let mut undecodable = vec![];
            for i in 0..self.n {
                if candidate_is_old[i] {
                    if box_contents[i] != 0 {
                        ok = false;
                        break;
                    }
                } else {
                    let expected_box_content = candidate_targets[i];
                    let actual_box_content = box_contents[i];
                    coordinate_expansion.push(i);
                    let injection_inverse = get_injection_inverse(
                        child_strategy.k(),
                        self.k,
                        &[(0, expected_box_content)],
                    );
                    let child_box_content = injection_inverse(actual_box_content);
                    if let Some(child_box_content) = child_box_content {
                        child_box_contents.push(child_box_content);
                    } else {
                        undecodable.push(i);
                        child_box_contents.push(0);
                    }
                }
            }
            if undecodable.len() > 1 {
                ok = false;
            }
            if ok {
                let child_options = child_strategy.decode(&child_box_contents);
                let parent_options: Vec<_> = child_options
                    .into_iter()
                    .map(|i| coordinate_expansion[i])
                    .collect();
                if undecodable.is_empty() {
                    options.extend(parent_options);
                } else if parent_options.contains(&undecodable[0]) {
                    options.insert(undecodable[0]);
                }
            }
        }
        if options.len() > 2 {
            return vec![];
        }
        options.into_iter().collect()
    }
}

fn small_strategy_cache_file_name(
    n: usize,
    k: u8,
    child_strategy: Option<Box<SmallStrategy>>,
) -> String {
    let mut s = child_strategy;
    let mut name = format!("cache/small_strategy_{}-{}", n, k);
    while let Some(t) = s {
        name = format!("{}_{}-{}", name, t.n, t.k);
        s = t.child_strategy;
    }
    name
}

fn get_outputs(strategy: &SmallStrategy, is_old: &[bool], child_outputs_cache: &[SmallStrategyOutput]) -> HashSet<SmallStrategyOutput> {
    let mut outputs = HashSet::new();
    let child_strategy = strategy.child_strategy.as_ref().unwrap();
    let target = strategy.target_map.get(is_old).unwrap();
    for output in child_outputs_cache {
        let mut expanded_box_contents = vec![0; strategy.n];
        let mut num_not_old_seen = 0;
        let mut last_box = None;
        for i in 0..strategy.n {
            if is_old[i] {
                expanded_box_contents[i] = 0;
            } else {
                let injection = get_injection(child_strategy.k(), strategy.k, &[(0, target[i])]);
                expanded_box_contents[i] = injection(output.box_contents[num_not_old_seen]);
                if num_not_old_seen == output.last_box {
                    last_box = Some(i);
                }
                num_not_old_seen += 1;
            }
        }
        for last_box_value in 0..strategy.k {
            expanded_box_contents[last_box.unwrap()] = last_box_value;
            outputs.insert(SmallStrategyOutput {
                is_old: is_old.to_owned(),
                box_contents: expanded_box_contents.clone(),
                last_box: last_box.unwrap(),
            });
        }
    }
    outputs
}

fn box_content_loss(
    box_contents: &Vec<u8>,
    all_outputs_by_box_contents: &HashMap<Vec<u8>, HashSet<SmallStrategyOutput>>,
) -> i64 {
    if let Some(outputs) = all_outputs_by_box_contents.get(box_contents) {
        let mut count_by_last_box: HashMap<_, _> = HashMap::new();
        for output in outputs {
            *count_by_last_box.entry(output.last_box).or_insert(0) += 1;
        }
        let mut counts = count_by_last_box.values().collect::<Vec<_>>();
        counts.sort();
        let mut loss = 0;
        for i in 0..(counts.len().max(2) - 2) {
            loss += counts[i] * counts[i];
        }
        loss
    } else {
        0
    }
}

fn update_target_map(
    strategy: &mut SmallStrategy,
    to_change: &SmallStrategyOutput,
    new_target: Vec<u8>,
    all_outputs_by_box_contents: &mut HashMap<Vec<u8>, HashSet<SmallStrategyOutput>>,
    loss: &mut i64,
    child_outputs_cache: &[SmallStrategyOutput]
) -> Vec<Vec<u8>> {
    for output in get_outputs(&strategy, &to_change.is_old, child_outputs_cache) {
        *loss -= box_content_loss(&output.box_contents, all_outputs_by_box_contents);
        assert!(all_outputs_by_box_contents
            .entry(output.box_contents.clone())
            .or_insert_with(|| HashSet::new())
            .remove(&output));
        *loss += box_content_loss(&output.box_contents, all_outputs_by_box_contents);
    }
    strategy
        .target_map
        .insert(to_change.is_old.clone(), new_target);
    let mut to_fix = vec![];
    for output in get_outputs(&strategy, &to_change.is_old, child_outputs_cache) {
        *loss -= box_content_loss(&output.box_contents, all_outputs_by_box_contents);
        assert!(all_outputs_by_box_contents
            .entry(output.box_contents.clone())
            .or_insert_with(|| HashSet::new())
            .insert(output.clone()));
        *loss += box_content_loss(&output.box_contents, all_outputs_by_box_contents);
        let outputs = all_outputs_by_box_contents
            .get(&output.box_contents)
            .unwrap();
        let last_boxes: HashSet<_> = outputs.iter().map(|output| output.last_box).collect();
        if last_boxes.len() <= 2 {
            continue;
        }
        to_fix.push(output.box_contents);
    }
    to_fix
}

pub fn get_small_strategy(
    n: usize,
    k: u8,
    child_strategy: Option<Box<SmallStrategy>>,
) -> SmallStrategy {
    let file_name = small_strategy_cache_file_name(n, k, child_strategy.clone());
    if let Ok(file) = File::open(&file_name) {
        let strategy: SmallStrategy = bincode::deserialize_from(file).unwrap();
        eprintln!("Loaded small strategy from {}", file_name);
        return strategy;
    }
    let mut strategy = SmallStrategy::new_random(n, k, child_strategy);
    let mut rng = StdRng::seed_from_u64(47);
    let mut steps = 0;
    let mut all_outputs_by_box_contents: HashMap<Vec<u8>, HashSet<SmallStrategyOutput>> =
        HashMap::new();
    let mut loss = 0;
    let child_outputs_cache: Vec<_> = strategy.child_strategy.as_ref().map(|child_strategy| child_strategy.get_all_outputs()).unwrap().into_iter().collect();
    for is_old in strategy.target_map.keys() {
        for output in get_outputs(&strategy, is_old, &child_outputs_cache) {
            all_outputs_by_box_contents
                .entry(output.box_contents.clone())
                .or_insert_with(|| HashSet::new())
                .insert(output);
        }
    }
    let mut to_fix: Queue<Vec<u8>> = Queue::new();
    let mut in_queue: HashSet<Vec<u8>> = HashSet::new();
    for (box_contents, outputs) in &all_outputs_by_box_contents {
        loss += box_content_loss(box_contents, &all_outputs_by_box_contents);
        let last_boxes: HashSet<_> = outputs.iter().map(|output| output.last_box).collect();
        if last_boxes.len() <= 2 {
            continue;
        }
        to_fix.add(box_contents.clone()).unwrap();
        in_queue.insert(box_contents.clone());
    }
    while let Ok(box_contents) = to_fix.remove() {
        in_queue.remove(&box_contents);
        steps += 1;
        let temperature = 1.0 / (1.0 + (10.0 * (steps as f64).ln()).sin()) - 0.5;

        let outputs: Vec<_> = all_outputs_by_box_contents
            .get(&box_contents)
            .unwrap()
            .iter()
            .cloned()
            .collect();
        if outputs.len() <= 2 {
            continue;
        }
        let to_change = outputs.choose(&mut rng).unwrap();

        let old_target = strategy.target_map.get(&to_change.is_old).unwrap().clone();
        let mut i = rng.gen_range(0..n);
        while to_change.is_old[i] {
            i = rng.gen_range(0..n);
        }
        let mut new_target = old_target.clone();
        new_target[i] = rng.gen_range(1..k);

        let old_loss = loss;
        let new_to_fix = update_target_map(
            &mut strategy,
            to_change,
            new_target.clone(),
            &mut all_outputs_by_box_contents,
            &mut loss,
            &child_outputs_cache
        );
        for new_to_fix in new_to_fix {
            if !in_queue.contains(&new_to_fix) {
                to_fix.add(new_to_fix.clone()).unwrap();
                in_queue.insert(new_to_fix);
            }
        }
        let delta = loss as f64 - old_loss as f64;
        let p = if delta <= 0.0 {
            1.0
        } else {
            (-delta / temperature).exp()
        };
        if !rng.gen_bool(p) {
            update_target_map(
                &mut strategy,
                to_change,
                old_target,
                &mut all_outputs_by_box_contents,
                &mut loss,
                &child_outputs_cache
            );
            assert!(loss == old_loss);
            to_fix.add(box_contents.clone()).unwrap();
            in_queue.insert(box_contents);
        }
    }

    let file = File::create(&file_name).unwrap();
    bincode::serialize_into(file, &strategy).unwrap();
    strategy
}
