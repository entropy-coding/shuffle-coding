use std::fmt::Debug;
use std::iter::empty;
use std::mem::size_of;
use std::ops::Deref;

use itertools::Itertools;
use lazy_static::lazy_static;
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64Mcg;

type Head = u64;
type TailElement = u8;

const HEAD_PREC: usize = size_of::<Head>() * 8;
const TAIL_PREC: usize = size_of::<TailElement>() * 8;
const MAX_MIN_HEAD: Head = 1 << (HEAD_PREC - TAIL_PREC);

/// Portable PRNG with "good" performance on statistical tests.
/// 16 bytes of state, 7 GB/s, compared to StdRng's 136 bytes of state, 1.5 GB/s
/// according to https://rust-random.github.io/book/guide-rngs.html.
/// Initialization is ~20x faster than StdRng.
/// Using StdRng made tests that create many messages annoyingly slow.
/// Fast message creation is tested by `tests::create_random_message_fast`.
type TailRng = Pcg64Mcg;

#[derive(Clone, Debug)]
pub enum TailGenerator {
    Random { rng: TailRng, seed: usize },
    Zeros,
    Empty,
}


impl TailGenerator {
    fn pop(&mut self) -> TailElement {
        match self {
            Self::Random { rng, .. } => { rng.gen() }
            Self::Zeros => 0,
            Self::Empty => panic!("Message exhausted whilst attempting decode.")
        }
    }

    fn reset_clone(&self) -> Self {
        match self {
            Self::Random { seed, .. } => Self::random(*seed),
            Self::Zeros => Self::Zeros,
            Self::Empty => Self::Empty,
        }
    }

    fn random(seed: usize) -> Self {
        Self::Random { rng: TailRng::seed_from_u64(seed as u64), seed }
    }
}

impl Iterator for TailGenerator {
    type Item = TailElement;
    fn next(&mut self) -> Option<Self::Item> { Some(self.pop()) }
}

#[derive(Clone, Debug)]
pub struct Tail {
    elements: Vec<TailElement>,
    generator: TailGenerator,
    num_generated: usize,
}

impl PartialEq for Tail {
    fn eq(&self, other: &Self) -> bool {
        let mut a = self.clone();
        let mut b = other.clone();
        a.normalize();
        b.normalize();

        a.elements == b.elements && a.num_generated == b.num_generated && match (&a.generator, &b.generator) {
            (TailGenerator::Random { seed, .. }, TailGenerator::Random { seed: s, .. }) => seed == s,
            (TailGenerator::Zeros, TailGenerator::Zeros) => true,
            (TailGenerator::Empty, TailGenerator::Empty) => true,
            _ => false,
        }
    }
}

impl Eq for Tail {}

impl Tail {
    fn new(elements: impl IntoIterator<Item=TailElement>, generator: TailGenerator) -> Self {
        Self { elements: elements.into_iter().collect(), generator, num_generated: 0 }
    }

    fn push(&mut self, element: TailElement) {
        self.elements.push(element);
    }

    fn pop(&mut self) -> TailElement {
        self.elements.pop().unwrap_or_else(|| {
            self.num_generated += 1;
            self.generator.pop()
        })
    }

    fn len_minus_generated(&self) -> isize { self.elements.len() as isize - self.num_generated as isize }

    fn normalize(&mut self) {
        if self.num_generated == 0 { return; }

        let mut generated = self.generator.reset_clone().
            take(self.num_generated).collect_vec();
        generated.reverse();
        let num_ungenerated = generated.iter().zip(self.elements.iter()).
            take_while(|(g, e)| g == e).count();

        self.elements.drain(..num_ungenerated);
        self.num_generated -= num_ungenerated;
        self.generator = self.generator.reset_clone();
        for _ in 0..self.num_generated {
            self.generator.pop();
        }
    }
}

#[derive(Clone, Debug)]
pub struct Message {
    pub head: Head,
    pub tail: Tail,
}

impl Message {
    /// Shifts bits between head and tail until min_head <= head < min_head << TAIL_PREC.
    fn renorm(&mut self, min_head: Head) {
        self.renorm_up(min_head);
        self.renorm_down(min_head);
    }

    /// Shifts bits from tail to head until min_head <= head.
    fn renorm_up(&mut self, min_head: Head) {
        while self.head < min_head {
            self.head = self.head << TAIL_PREC | self.tail.pop() as Head
        }
    }

    /// Shifts bits from head to tail until head < min_head << TAIL_PREC.
    fn renorm_down(&mut self, min_head: Head) {
        loop {
            let new_head = self.head >> TAIL_PREC;
            if new_head < min_head { break; }
            self.tail.push(self.head as TailElement);
            self.head = new_head;
        }
    }

    pub fn flatten(mut self) -> Tail {
        self.renorm_down(1);
        let mut tail = self.tail;
        tail.push(self.head as TailElement);
        tail
    }

    pub fn unflatten(tail: Tail) -> Message {
        Message { head: 0, tail }
    }

    /// Actual number of bits to be sent/stored.
    pub fn bits(&self) -> usize {
        TAIL_PREC * self.clone().flatten().elements.len()
    }

    /// Precise virtual message length in bits where the head is counted as log(head) and
    /// generated tail is subtracted. Fractional in general and can be negative.
    /// The increase in virtual bits when pushing a symbol is its info content under the used codec.
    pub fn virtual_bits(&self) -> f64 {
        let mut clone: Message;
        let message = if self.head > 1 << 32 { self } else {
            clone = self.clone();
            // Avoid inaccuracy for small messages:
            clone.renorm_up(MAX_MIN_HEAD);
            &clone
        };
        (message.head as f64).log2() + (TAIL_PREC as isize * message.tail.len_minus_generated()) as f64
    }

    pub fn random(seed: usize) -> Message {
        let tail = Tail::new(empty(), TailGenerator::random(seed));
        let mut m = Message { head: 1, tail };
        m.renorm_up(MAX_MIN_HEAD);
        m
    }

    pub fn zeros() -> Self {
        Self { head: MAX_MIN_HEAD, tail: Tail::new(empty(), TailGenerator::Zeros) }
    }

    #[allow(unused)]
    pub fn empty() -> Self {
        Self { head: MAX_MIN_HEAD, tail: Tail::new(empty(), TailGenerator::Empty) }
    }
}

impl PartialEq for Message {
    fn eq(&self, other: &Self) -> bool {
        let mut m = self.clone();
        m.renorm(MAX_MIN_HEAD);
        let mut o = other.clone();
        o.renorm(MAX_MIN_HEAD);
        m.tail == o.tail && m.head == o.head
    }
}

pub trait Symbol: Clone + Debug + Eq {}

impl<T: Clone + Debug + Eq> Symbol for T {}

pub trait Codec: Clone {
    type Symbol: Symbol;
    fn push(&self, m: &mut Message, x: &Self::Symbol);
    fn pop(&self, m: &mut Message) -> Self::Symbol;
    /// Code length for the given symbol in bits if deterministic and known, None otherwise.
    fn bits(&self, x: &Self::Symbol) -> Option<f64>;

    fn sample(&self, seed: usize) -> Self::Symbol {
        self.pop(&mut Message::random(seed))
    }

    fn samples(&self, len: usize, seed: usize) -> Vec<Self::Symbol> {
        IID::new(self.clone(), len).sample(seed)
    }

    /// Any implementation should pass this test for any seed and valid symbol x.
    fn test_invertibility(&self, x: &Self::Symbol, initial: &Message) -> CodecTestResults {
        let m = &mut initial.clone();
        let enc_sec = timeit_loops!(1, { self.push(m, x) });
        let bits = m.bits();
        let amortized_bits = m.virtual_bits() - initial.virtual_bits();
        assert!(bits as f64 >= amortized_bits);
        let mut decoded: Option<Self::Symbol> = None;
        let dec_sec = timeit_loops!(1, { decoded = Some(self.pop(m)) });
        assert_eq!(x, &decoded.unwrap());
        assert_eq!(initial, m);
        assert_eq!(initial, &Message::unflatten(m.clone().flatten()));
        CodecTestResults { bits, amortized_bits, enc_sec, dec_sec }
    }

    /// Any implementation should pass this test for any seed and valid symbol x.
    fn test(&self, x: &Self::Symbol, initial: &Message) -> CodecTestResults {
        let out = self.test_invertibility(x, initial);
        if let Some(amortized_bits) = self.bits(x) {
            assert_bits_eq(amortized_bits, out.amortized_bits);
        }
        out
    }

    /// Any implementation should pass this test for any num_samples.
    fn test_on_samples(&self, num_samples: usize) -> Vec<f64> {
        (0..num_samples).map(|seed| self.test(&self.sample(seed), &Message::random(seed))).map(|r| r.amortized_bits).collect()
    }
}

/// Codec with a uniform distribution. All codes have the same length.
pub trait UniformCodec: Codec {
    /// Codec length for all symbols in bits.
    fn uni_bits(&self) -> f64;
}

pub struct CodecTestResults {
    pub bits: usize,
    pub amortized_bits: f64,
    pub enc_sec: f64,
    pub dec_sec: f64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Uniform {
    pub size: Head,
    max_min_head_div_size: Head,
}

impl Codec for Uniform {
    type Symbol = usize;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        assert!(*x < self.size as usize);
        m.renorm(self.max_min_head_div_size);
        m.head = self.size * m.head + *x as Head
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        m.renorm(self.size * self.max_min_head_div_size);
        let x = m.head % self.size;
        m.head /= self.size;
        x as Self::Symbol
    }

    fn bits(&self, _: &Self::Symbol) -> Option<f64> { Some(self.uni_bits()) }
}

impl UniformCodec for Uniform {
    fn uni_bits(&self) -> f64 {
        (self.size as f64).log2()
    }
}

impl Uniform {
    /// Uniform.max() would have inaccurate bits() values for larger sizes,
    /// so we disallow them to simplify testing.
    pub const MAX_SIZE: usize = (MAX_MIN_HEAD >> 10) as usize;

    pub fn new(size: usize) -> Self {
        assert!(size <= Self::MAX_SIZE);
        let max_min_head_div_size = MAX_MIN_HEAD / size as Head;
        Self { size: size as Head, max_min_head_div_size }
    }

    pub fn max() -> &'static Self {
        lazy_static! {static ref C: Uniform = Uniform::new(Uniform::MAX_SIZE);}
        &C
    }
}

/// OPTIMIZE: revert to fused DistributionCodec-style implementation
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Categorical {
    /// None if the probability mass for a symbol is 0.
    pub codecs: Vec<Option<Uniform>>,
    pub cummasses: Vec<usize>,
    pub cumcodec: Uniform,
}

impl Codec for Categorical {
    type Symbol = usize;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        let i = self.codecs[*x].as_ref().unwrap().pop(m);
        let cf = self.cummasses[*x] + i;
        self.cumcodec.push(m, &cf);
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        let cf = self.cumcodec.pop(m);
        let x = self.cummasses.partition_point(|&c| c <= cf) - 1;
        let codec = self.codecs[x].as_ref().unwrap();
        codec.push(m, &(cf - self.cummasses[x]));
        x
    }

    fn bits(&self, x: &Self::Symbol) -> Option<f64> {
        Some(self.cumcodec.uni_bits() - self.codecs[*x].as_ref().unwrap().uni_bits())
    }
}

impl Categorical {
    pub fn new(masses: &Vec<usize>) -> Self {
        let cummasses = masses.iter().scan(0, |acc, &x| {
            let out = Some(*acc);
            *acc += x;
            out
        }).collect();
        let codecs = masses.iter().map(|&m| if m == 0 { None } else { Some(Uniform::new(m)) }).collect();
        let cumcodec = Uniform::new(masses.iter().sum());
        Self { codecs, cummasses, cumcodec }
    }

    pub fn masses(&self) -> Vec<usize> {
        self.codecs.iter().map(|c| if let Some(c) = c { c.size as usize } else { 0 }).collect()
    }

    pub fn mass(&self, x: usize) -> usize {
        if let Some(c) = &self.codecs[x] { c.size as usize } else { 0 }
    }

    pub fn prob(&self, x: usize) -> f64 {
        self.mass(x) as f64 / self.cumcodec.size as f64
    }

    pub fn entropy(&self) -> f64 {
        (0..self.codecs.len()).map(|x| {
            let p = self.prob(x);
            if p == 0. { 0. } else { -p.log2() * p }
        }).sum::<f64>()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Bernoulli {
    pub categorical: Categorical,
}

impl Codec for Bernoulli {
    type Symbol = bool;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        self.categorical.push(m, &(*x as usize));
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        self.categorical.pop(m) != 0
    }

    fn bits(&self, x: &Self::Symbol) -> Option<f64> {
        self.categorical.bits(&(*x as usize))
    }
}

impl Bernoulli {
    pub fn prob(&self) -> f64 {
        self.categorical.prob(1)
    }

    #[allow(unused)]
    pub fn from_prob(prob: f64, norm: usize) -> Self {
        Self::new((prob * norm as f64) as usize, norm)
    }

    pub fn new(mass: usize, norm: usize) -> Self {
        assert!(mass <= norm);
        Self { categorical: Categorical::new(&vec![norm - mass, mass]) }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VecCodec<C: Codec> {
    pub codecs: Vec<C>,
}

impl<C: Codec> Codec for VecCodec<C> {
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

impl<C: UniformCodec> UniformCodec for VecCodec<C> {
    fn uni_bits(&self) -> f64 {
        self.codecs.iter().map(|c| c.uni_bits()).sum()
    }
}

impl<C: Codec> VecCodec<C> {
    pub fn new(codecs: impl IntoIterator<Item=C>) -> Self { Self { codecs: codecs.into_iter().collect() } }
}

pub fn assert_bits_eq(expected_bits: f64, bits: f64) {
    assert_bits_close(expected_bits, bits, 1e-5);
}

pub fn assert_bits_close(expected_bits: f64, bits: f64, tol: f64) {
    let mismatch = (bits - expected_bits).abs() / expected_bits.abs().max(1.);
    assert!(mismatch < tol, "Expected {} bits, but got {} bits.", expected_bits, bits);
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

#[derive(Clone, Debug)]
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

pub trait Distribution: Clone {
    type Symbol: Symbol;

    fn norm(&self) -> usize;
    fn pmf(&self, x: &Self::Symbol) -> usize;
    fn cdf(&self, x: &Self::Symbol, i: usize) -> usize;
    fn icdf(&self, cf: usize) -> (Self::Symbol, usize);
}

#[derive(Clone, Debug)]
pub struct DistCodec<D: Distribution>(pub D);

impl<D: Distribution> Codec for DistCodec<D> {
    type Symbol = D::Symbol;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        let i = self.codec(x).pop(m);
        let some_element = self.0.cdf(x, i);
        self.cumcodec().push(m, &some_element);
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        let cf = self.cumcodec().pop(m);
        let (x, i) = self.0.icdf(cf);
        self.codec(&x).push(m, &i);
        x
    }

    fn bits(&self, x: &Self::Symbol) -> Option<f64> {
        Some(self.cumcodec().uni_bits() - self.codec(x).uni_bits())
    }
}

impl<D: Distribution> DistCodec<D> {
    fn cumcodec(&self) -> Uniform {
        Uniform::new(self.0.norm())
    }

    fn codec(&self, x: &D::Symbol) -> Uniform {
        Uniform::new(self.0.pmf(x))
    }
}

#[derive(Clone, Debug)]
pub struct Benford {
    pub bits: Uniform,
}

impl Codec for Benford {
    type Symbol = usize;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        let bits = Self::get_bits(x);
        assert!(bits < self.bits.size as usize);
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

impl Benford {
    pub fn new(excl_max_bits: usize) -> Self {
        assert!(excl_max_bits <= size_of::<<Self as Codec>::Symbol>() * 8);
        Self { bits: Uniform::new(excl_max_bits + 1) }
    }

    pub fn max() -> &'static Self {
        lazy_static! {static ref C: Benford = Benford::new(47);}
        &C
    }

    pub fn get_bits(x: &usize) -> usize {
        size_of::<<Self as Codec>::Symbol>() * 8 - x.leading_zeros() as usize
    }
}

#[cfg(test)]
pub mod tests {
    use std::iter::repeat;

    use super::*;

    fn assert_entropy_eq(expected_entropy: f64, entropy: f64) {
        assert_bits_close(expected_entropy, entropy, 0.02);
    }

    #[test]
    fn create_random_messages_fast() {
        let sec = timeit_loops!(100000, { Message::random(0); });
        assert!(sec < 5e-6, "{}s is too slow for creating a random message.", sec);
    }

    const NUM_SAMPLES: usize = 1000;

    #[test]
    fn dists() {
        let c = Categorical::new(&vec![0, 1, 2, 3, 0, 0, 1, 0]);
        assert_entropy_eq(c.entropy(), c.test_on_samples(NUM_SAMPLES).iter().sum::<f64>() / NUM_SAMPLES as f64);
        test_bernoulli(0.2);
        test_bernoulli(0.);
        test_bernoulli(1.);
        Uniform::new(1 << 28).test_on_samples(NUM_SAMPLES);
        IID::new(Uniform::new(1 << 28), 2).test_on_samples(NUM_SAMPLES);
        VecCodec::new(repeat(Uniform::new(1 << 28)).take(2)).test_on_samples(NUM_SAMPLES);
    }

    fn test_bernoulli(prob: f64) {
        let c = Bernoulli::from_prob(prob, 1 << 28);
        assert_entropy_eq(c.categorical.entropy(), c.test_on_samples(NUM_SAMPLES).iter().sum::<f64>() / NUM_SAMPLES as f64);
    }

    #[test]
    fn truncated_benford() {
        let c = Benford::new(8);
        for i in 0..255 {
            c.test(&i, &Message::random(0));
        }
    }
}
