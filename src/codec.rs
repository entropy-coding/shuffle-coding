//! Implements various elementary ANS codecs.
use std::cell::RefCell;
use std::fmt::Debug;
use std::hash::Hash;
use std::mem;
use std::ops::{Deref, DerefMut};

use itertools::Itertools;
use lazy_static::lazy_static;

pub use crate::ans::*;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Uniform {
    pub size: usize,
}

impl Distribution for Uniform {
    type Symbol = usize;

    fn norm(&self) -> usize { self.size }
    fn pmf(&self, _x: &Self::Symbol) -> usize { 1 }
    fn cdf(&self, x: &Self::Symbol, i: usize) -> usize {
        assert_eq!(i, 0);
        *x
    }

    fn icdf(&self, cf: usize) -> (Self::Symbol, usize) {
        (cf as Self::Symbol, 0)
    }
}

impl Uniform {
    pub fn new(size: usize) -> Self {
        assert!(size <= MAX_SIZE);
        Self { size }
    }

    pub fn max() -> &'static Self {
        lazy_static! {static ref C: Uniform = Uniform::new(MAX_SIZE);}
        &C
    }
}

impl UniformCodec for Uniform {
    fn uni_bits(&self) -> f64 {
        (self.size as f64).log2()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Categorical {
    /// None if the probability mass for a symbol is 0.
    pub masses: Vec<usize>,
    pub cummasses: Vec<usize>,
    pub norm: usize,
}

impl Distribution for Categorical {
    type Symbol = usize;

    fn norm(&self) -> usize { self.norm }
    fn pmf(&self, x: &Self::Symbol) -> usize { self.masses[*x] }
    fn cdf(&self, x: &Self::Symbol, i: usize) -> usize { self.cummasses[*x] + i }
    fn icdf(&self, cf: usize) -> (Self::Symbol, usize) {
        let x = self.cummasses.partition_point(|&c| c <= cf) - 1;
        (x as Self::Symbol, cf - self.cummasses[x])
    }
}

impl Categorical {
    pub fn new(masses: Vec<usize>) -> Self {
        let cummasses = masses.iter().scan(0, |acc, &x| {
            let out = Some(*acc);
            *acc += x;
            out
        }).collect();
        let norm = masses.iter().sum();
        Self { masses, cummasses, norm }
    }

    pub fn prob(&self, x: usize) -> f64 {
        self.masses[x] as f64 / self.norm() as f64
    }

    pub fn entropy(&self) -> f64 {
        (0..self.masses.len()).map(|x| {
            let p = self.prob(x);
            if p == 0. { 0. } else { -p.log2() * p }
        }).sum::<f64>()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Bernoulli {
    pub categorical: Categorical,
}

impl Distribution for Bernoulli {
    type Symbol = bool;

    fn norm(&self) -> usize {
        self.categorical.norm
    }

    fn pmf(&self, x: &Self::Symbol) -> usize {
        self.categorical.pmf(&(*x as usize))
    }

    fn cdf(&self, x: &Self::Symbol, i: usize) -> usize {
        self.categorical.cdf(&(*x as usize), i)
    }

    fn icdf(&self, cf: usize) -> (Self::Symbol, usize) {
        let (x, i) = self.categorical.icdf(cf);
        (x != 0, i)
    }
}

impl Bernoulli {
    pub fn prob(&self) -> f64 {
        self.categorical.prob(1)
    }

    pub fn new(mass: usize, norm: usize) -> Self {
        assert!(mass <= norm);
        Self { categorical: Categorical::new(vec![norm - mass, mass]) }
    }
}

pub trait OrdSymbol: Symbol + Ord + Hash {}
impl<T: Symbol + Ord + Hash> OrdSymbol for T {}

/// Categorical allowing insertion and removal of mass.
/// Insert, remove, pmf, cdf and icdf all have a runtime of O(log #symbols).
/// Implemented via an order statistic tree.
#[derive(Clone, Debug, Default, Eq)]
pub struct MutCategorical<S: OrdSymbol + Default = usize> {
    branches: Option<Box<(Self, Self)>>,
    count: usize,
    split: S,
    depth: usize,
}

impl<S: OrdSymbol + Default> Distribution for MutCategorical<S> {
    type Symbol = S;

    fn norm(&self) -> usize {
        self.count
    }

    fn pmf(&self, x: &Self::Symbol) -> usize {
        if let Some(branches) = &self.branches {
            let (left, right) = branches.deref();
            if x < &self.split { left } else { right }.pmf(x)
        } else {
            if x == &self.split { self.count } else { 0 }
        }
    }

    fn cdf(&self, x: &Self::Symbol, i: usize) -> usize {
        if let Some(branches) = &self.branches {
            let (left, right) = branches.deref();
            if x < &self.split { left.cdf(x, i) } else { left.count + right.cdf(x, i) }
        } else {
            assert_eq!(&self.split, x, "Symbol {x:?} not found in distribution.");
            assert!(i < self.count);
            i
        }
    }

    fn icdf(&self, cf: usize) -> (Self::Symbol, usize) {
        if let Some(branches) = &self.branches {
            let (left, right) = branches.deref();
            if cf < left.count { left.icdf(cf) } else { right.icdf(cf - left.count) }
        } else {
            assert!(cf < self.count);
            (self.split.clone(), cf)
        }
    }
}

impl<S: OrdSymbol + Default> MutDistribution for MutCategorical<S> {
    fn insert(&mut self, x: Self::Symbol, mass: usize) {
        let is_left = x < self.split;
        if let Some(branches) = &mut self.branches {
            assert_ne!(self.count, 0);
            let (left, right) = branches.deref_mut();
            let tree = if is_left { left } else { right };
            tree.insert(x, mass);
        } else if x != self.split {
            if self.count == 0 {
                self.split = x;
            } else {
                let new = Self::leaf(x.clone(), mass);
                let prev = Self::leaf(self.split.clone(), self.count);
                self.branches = Some(Box::new(if is_left {
                    (new, prev)
                } else {
                    self.split = x;
                    (prev, new)
                }));
            }
        }
        self.count += mass;
        self.rebalance_and_update_depth();
    }

    fn remove(&mut self, x: &Self::Symbol, mass: usize) {
        assert!(mass <= self.count);
        self.count -= mass;

        if let Some(branches) = &mut self.branches {
            let (left, right) = branches.deref_mut();
            let is_left = x < &self.split;
            let tree = if is_left { left } else { right };
            tree.remove(x, mass);
            if tree.count == 0 {
                let (left, right) = branches.deref_mut();
                *self = mem::take(if is_left { right } else { left });
            } else {
                self.rebalance_and_update_depth();
            }
            assert_ne!(self.count, 0);
        }
    }
}

#[derive(Clone, Debug)]
pub struct MutCategoricalIter<'a, S: OrdSymbol + Default = usize> {
    stack: Vec<&'a MutCategorical<S>>,
}

impl<'a, S: OrdSymbol + Default> MutCategoricalIter<'a, S> {
    fn push_left_edge(&mut self, mut node: &'a MutCategorical<S>) {
        loop {
            if node.count == 0 {
                assert!(node.branches.is_none());
                break;
            }
            self.stack.push(node);
            if let Some(branch) = &node.branches {
                node = &branch.deref().0;
            } else {
                break;
            }
        }
    }
}

impl<'a, S: OrdSymbol + Default> Iterator for MutCategoricalIter<'a, S> {
    type Item = (S, usize);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(node) = self.stack.pop() {
            if let Some(branch) = &node.branches {
                self.push_left_edge(&branch.deref().1);
            } else {
                assert_ne!(node.count, 0);
                return Some((node.split.clone(), node.count.clone()));
            }
        }
        None
    }
}

impl<S: OrdSymbol + Default> PartialEq for MutCategorical<S> {
    fn eq(&self, other: &Self) -> bool {
        self.iter().eq(other.iter())
    }
}

impl<S: OrdSymbol + Default> MutCategorical<S> {
    fn update_depth(&mut self) {
        self.depth = if let Some(branches) = &self.branches {
            let (left, right) = branches.deref();
            left.depth.max(right.depth) + 1
        } else { 0 };
    }

    fn update_depth_and_count(&mut self) {
        self.update_depth();
        let (l, r) = self.branches_mut();
        self.count = l.count + r.count;
    }

    fn rebalance_and_update_depth(&mut self) {
        let balance = self.balance_factor();
        if balance > 1 {
            let left = self.left();
            if left.balance_factor() < 0 {
                left.rotate_left();
            }
            self.rotate_right();
        } else if balance < -1 {
            let right = self.right();
            if right.balance_factor() > 0 {
                right.rotate_right();
            }
            self.rotate_left();
        } else {
            self.update_depth();
        }
    }


    fn balance_factor(&self) -> isize {
        if let Some(branches) = &self.branches {
            let (left, right) = branches.deref();
            left.depth as isize - right.depth as isize
        } else { 0 }
    }

    fn rotate_left(&mut self) { self.rotate(false) }
    fn rotate_right(&mut self) { self.rotate(true) }
    fn rotate(&mut self, right: bool) {
        // Variable names are according to the case of right rotation (right == true):
        let get_left = if right { Self::left } else { Self::right };
        let get_right = if right { Self::right } else { Self::left };
        let left = get_left(self);
        let left_right = mem::take(get_right(left));
        let mut left = mem::replace(left, left_right);
        let left_right = get_right(&mut left);
        *left_right = mem::take(self);
        left_right.update_depth_and_count();
        *self = left;
        self.update_depth_and_count();
    }

    fn left(&mut self) -> &mut Self {
        &mut self.branches_mut().0
    }

    fn right(&mut self) -> &mut Self {
        &mut self.branches_mut().1
    }

    fn branches_mut(&mut self) -> &mut (MutCategorical<S>, MutCategorical<S>) {
        self.branches.as_mut().unwrap().deref_mut()
    }

    #[allow(unused)]
    fn iter(&self) -> MutCategoricalIter<S> {
        let mut iter = MutCategoricalIter { stack: Vec::new() };
        iter.push_left_edge(self);
        iter
    }

    fn leaf(x: S, count: usize) -> Self {
        Self { branches: None, count, split: x, depth: 0 }
    }

    pub fn new(masses: impl IntoIterator<Item=(S, usize)>) -> Self {
        let mut out: Option<Self> = None;
        for (x, count) in masses {
            if let Some(out) = &mut out {
                out.insert(x, count);
            } else {
                out = Some(Self::leaf(x, count));
            }
        }
        out.unwrap_or_else(Self::default)
    }
}

/// Codec for a vector of independent symbols with distributions of the same type.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Independent<C: Codec> {
    pub codecs: Vec<C>,
}

impl<C: Codec> Codec for Independent<C> {
    type Symbol = Vec<C::Symbol>;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        assert_eq!(x.len(), self.codecs.len());
        for (x, codec) in x.iter().zip(&self.codecs).rev() {
            codec.push(m, &x)
        }
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        self.codecs.iter().map(|codec| codec.pop(m)).collect()
    }

    fn bits(&self, x: &Self::Symbol) -> Option<f64> {
        let mut total = 0.;
        for (c, x) in self.codecs.iter().zip_eq(x) {
            total += c.bits(x)?;
        }
        Some(total)
    }
}

impl<C: UniformCodec> UniformCodec for Independent<C> {
    fn uni_bits(&self) -> f64 {
        self.codecs.iter().map(|c| c.uni_bits()).sum()
    }
}

impl<C: Codec> Independent<C> {
    pub fn new(codecs: impl IntoIterator<Item=C>) -> Self { Self { codecs: codecs.into_iter().collect() } }
}

/// Codec with independent and identically distributed symbols.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IID<C: Codec> {
    pub item: C,
    pub len: usize,
}

impl<C: Codec> Codec for IID<C> {
    type Symbol = Vec<C::Symbol>;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        assert_eq!(x.len(), self.len);
        for e in x.iter().rev() {
            self.item.push(m, e)
        }
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        (0..self.len).map(|_| self.item.pop(m)).collect()
    }

    fn bits(&self, x: &Self::Symbol) -> Option<f64> {
        let mut total = 0.;
        for x in x.iter() {
            total += self.item.bits(x)?;
        }
        Some(total)
    }
}

impl<C: UniformCodec> UniformCodec for IID<C> {
    fn uni_bits(&self) -> f64 {
        self.len as f64 * self.item.uni_bits()
    }
}

impl<C: Codec> IID<C> {
    pub fn new(item: C, len: usize) -> Self { Self { item, len } }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ConstantCodec<T: Symbol>(pub T);

impl<T: Symbol> Deref for ConstantCodec<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target { &self.0 }
}

impl<T: Default + Symbol> Default for ConstantCodec<T> {
    fn default() -> Self { Self(T::default()) }
}

impl<T: Symbol> Codec for ConstantCodec<T> {
    type Symbol = T;
    fn push(&self, _: &mut Message, x: &Self::Symbol) { assert_eq!(x, &self.0); }
    fn pop(&self, _: &mut Message) -> Self::Symbol { self.0.clone() }
    fn bits(&self, _: &Self::Symbol) -> Option<f64> { Some(self.uni_bits()) }
}

impl<T: Symbol> UniformCodec for ConstantCodec<T> {
    fn uni_bits(&self) -> f64 { 0. }
}

impl<A: Codec, B: Codec> Codec for (A, B) {
    type Symbol = (A::Symbol, B::Symbol);

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        self.1.push(m, &x.1);
        self.0.push(m, &x.0);
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        let a = self.0.pop(m);
        (a, self.1.pop(m))
    }

    fn bits(&self, x: &Self::Symbol) -> Option<f64> {
        Some(self.0.bits(&x.0)? + self.1.bits(&x.1)?)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EnumCodec<A: Codec, B: Codec<Symbol=A::Symbol>> {
    A(A),
    B(B),
}

impl<A: Codec, B: Codec<Symbol=A::Symbol>> Codec for EnumCodec<A, B> {
    type Symbol = A::Symbol;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        match self {
            Self::A(a) => a.push(m, x),
            Self::B(b) => b.push(m, x),
        }
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        match self {
            Self::A(a) => a.pop(m),
            Self::B(b) => b.pop(m),
        }
    }

    fn bits(&self, x: &Self::Symbol) -> Option<f64> {
        match self {
            Self::A(a) => a.bits(x),
            Self::B(b) => b.bits(x),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OptionCodec<C: Codec> {
    pub is_some: Bernoulli,
    pub some: C,
}

impl<C: Codec> Codec for OptionCodec<C> {
    type Symbol = Option<C::Symbol>;
    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        if let Some(x) = x {
            self.some.push(m, x);
        }
        self.is_some.push(m, &x.is_some());
    }
    fn pop(&self, m: &mut Message) -> Self::Symbol {
        if self.is_some.pop(m) {
            Some(self.some.pop(m))
        } else {
            None
        }
    }

    fn bits(&self, x: &Self::Symbol) -> Option<f64> {
        let is_some_bits = self.is_some.bits(&x.is_some());
        if let Some(x) = x {
            Some(is_some_bits? + self.some.bits(x)?)
        } else {
            is_some_bits
        }
    }
}

pub trait MutDistribution: Distribution {
    fn insert(&mut self, x: Self::Symbol, mass: usize);
    fn remove(&mut self, x: &Self::Symbol, mass: usize);

    fn remove_all(&mut self, x: &Self::Symbol) -> usize {
        let count = self.pmf(x);
        self.remove(x, count);
        assert_eq!(self.pmf(x), 0);
        count
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LogUniform {
    pub bits: Uniform,
}

impl Codec for LogUniform {
    type Symbol = usize;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        let bits = Self::get_bits(x);
        assert!(bits < self.bits.size);
        if bits != 0 {
            let size = 1 << (bits - 1);
            Uniform::new(size).push(m, &(x & !size));
        }
        self.bits.push(m, &bits);
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        let bits = self.bits.pop(m);
        if bits == 0 { 0 } else {
            let size = 1 << (bits - 1);
            Uniform::new(size).pop(m) | size
        }
    }

    fn bits(&self, x: &Self::Symbol) -> Option<f64> {
        let bits = Self::get_bits(x);
        Some(self.bits.uni_bits() + if bits == 0 { 0. } else {
            let size = 1 << (bits - 1);
            Uniform::new(size).uni_bits()
        })
    }
}

impl LogUniform {
    pub fn new(excl_max_bits: usize) -> Self {
        assert!(excl_max_bits <= mem::size_of::<<Self as Codec>::Symbol>() * 8);
        let size = excl_max_bits + 1;
        Self { bits: Uniform::new(size) }
    }

    pub fn max() -> &'static Self {
        lazy_static! {static ref C: LogUniform = LogUniform::new(47);}
        &C
    }

    pub fn get_bits(x: &usize) -> usize {
        mem::size_of::<<Self as Codec>::Symbol>() * 8 - x.leading_zeros() as usize
    }
}

impl<C: Codec> Codec for RefCell<C> {
    type Symbol = C::Symbol;

    fn push(&self, m: &mut Message, x: &Self::Symbol) { self.borrow().push(m, x) }
    fn pop(&self, m: &mut Message) -> Self::Symbol { self.borrow().pop(m) }
    fn bits(&self, x: &Self::Symbol) -> Option<f64> { self.borrow().bits(x) }
}

#[cfg(test)]
pub mod tests {
    use itertools::repeat_n;

    use crate::benchmark::test_and_print;

    use super::*;

    fn assert_entropy_eq(expected_entropy: f64, entropy: f64) {
        assert_bits_close(expected_entropy, entropy, 0.02);
    }

    pub fn test_and_print_vec<C: Codec>(codecs: impl IntoIterator<Item=C>, symbols: &Vec<C::Symbol>, initial: &Message) -> CodecTestResults {
        test_and_print(&Independent::new(codecs), symbols, initial)
    }

    #[test]
    fn create_random_messages_fast() {
        let sec = timeit_loops!(100000, { Message::random(0); });
        assert!(sec < 5e-6, "{}s is too slow for creating a random message.", sec);
    }

    const NUM_SAMPLES: usize = 1000;

    #[test]
    fn dists() {
        let masses = vec![0, 1, 2, 3, 0, 0, 1, 0];
        let c = Categorical::new(masses);
        assert_entropy_eq(c.entropy(), c.test_on_samples(NUM_SAMPLES).iter().sum::<f64>() / NUM_SAMPLES as f64);
        test_bernoulli(2, 10);
        test_bernoulli(0, 10);
        test_bernoulli(10, 10);
        Uniform::new(1 << 28).test_on_samples(NUM_SAMPLES);
        IID::new(Uniform::new(1 << 28), 2).test_on_samples(NUM_SAMPLES);
        Independent::new(repeat_n(Uniform::new(1 << 28), 2)).test_on_samples(NUM_SAMPLES);
    }

    fn test_bernoulli(mass: usize, norm: usize) {
        let c = Bernoulli::new(mass, norm);
        assert_entropy_eq(c.categorical.entropy(), c.test_on_samples(NUM_SAMPLES).iter().sum::<f64>() / NUM_SAMPLES as f64);
    }

    #[test]
    fn truncated_benford() {
        let c = LogUniform::new(8);
        for i in 0..255 {
            c.test(&i, &Message::random(0));
        }
    }

    #[test]
    fn empty_mut_categorical() {
        let mut c = MutCategorical::default();
        assert_eq!(c.iter().collect_vec(), vec![]);
        c.insert(5, 1);
        assert_eq!(c.iter().collect_vec(), vec![(5, 1)]);
        c.remove(&5, 1);
        assert_eq!(c.iter().collect_vec(), vec![]);
        c.insert(3, 1);
        assert_eq!(c.iter().collect_vec(), vec![(3, 1)]);
        c.insert(2, 5);
        c.insert(7, 2);
        assert_eq!(c.iter().collect_vec(), vec![(2, 5), (3, 1), (7, 2)]);
        c.remove(&3, 1);
        assert_eq!(c.iter().collect_vec(), vec![(2, 5), (7, 2)]);
    }

    #[test]
    fn mut_categorical() {
        let mut dist = MutCategorical::new(vec![0, 1, 2, 4, 0, 0, 1, 0, 4].into_iter().map(|x| (x, 1)));
        assert_eq!(dist.norm(), 9);

        assert_eq!(dist.pmf(&0), 4);
        assert_eq!(dist.pmf(&1), 2);
        assert_eq!(dist.pmf(&2), 1);
        assert_eq!(dist.pmf(&3), 0);
        assert_eq!(dist.pmf(&4), 2);

        assert_eq!(dist.iter().collect_vec(), [(0, 4), (1, 2), (2, 1), (4, 2)]);

        assert_eq!(dist.cdf(&0, 0), 0);
        assert_eq!(dist.cdf(&1, 0), 4);
        assert_eq!(dist.cdf(&2, 0), 6);
        assert_eq!(dist.cdf(&4, 0), 7);

        dist.test_on_samples(50);

        dist.remove(&0, 2);

        assert_eq!(dist.norm(), 7);

        assert_eq!(dist.pmf(&0), 2);
        assert_eq!(dist.pmf(&1), 2);

        assert_eq!(dist.cdf(&0, 0), 0);
        assert_eq!(dist.cdf(&1, 0), 2);
        assert_eq!(dist.cdf(&2, 0), 4);
        assert_eq!(dist.cdf(&4, 0), 5);

        dist.test_on_samples(50);

        dist.remove(&0, 2);
        dist.test_on_samples(50);
        dist.remove(&4, 2);
        assert_eq!(dist.cdf(&1, 0), 0);
        assert_eq!(dist.cdf(&2, 0), 2);
        dist.test_on_samples(50);
    }

    #[test]
    fn test_fail() {
        let dist = MutCategorical::new(vec![(0, 1), (1, 2), (2, 2)]);
        assert_eq!(dist.norm(), 5);
        for seed in 0..500 {
            dist.test(&2, &Message::random(seed));
        }
        dist.test_on_samples(50);
    }
}