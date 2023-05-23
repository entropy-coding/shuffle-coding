use std::marker::PhantomData;

use crate::ans::{Codec, DistCodec, Distribution, Message};
use crate::permutable::{Automorphisms, Cell, GroupPermutable, Orbit, Partial, Permutable, Permutation};
use crate::shuffle_ans::Unordered;

#[derive(Clone)]
pub struct InterleavedShuffleCodec<P, D, DFromLen, Pa>
    where P: Partial + GroupPermutable,
          P::Complete: GroupPermutable,
          D: Codec<Symbol=P::Diff>,
          DFromLen: Fn(usize) -> D + Clone,
          Pa: Partitioner<P> {
    pub partial: PartialShuffleCodec<P, D, DFromLen, Pa>,
}

impl<P, D, DFromLen, Pa> Codec for InterleavedShuffleCodec<P, D, DFromLen, Pa> where
    P: Partial + GroupPermutable,
    P::Complete: GroupPermutable,
    D: Codec<Symbol=P::Diff>,
    DFromLen: Fn(usize) -> D + Clone,
    Pa: Partitioner<P> {
    type Symbol = Unordered<P::Complete>;

    fn push(&self, m: &mut Message, Unordered(x): &Self::Symbol) {
        self.partial.push(m, &Unordered(P::from_complete(x.clone())));
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        Unordered(self.partial.pop(m).0.into_complete())
    }

    fn bits(&self, Unordered(x): &Self::Symbol) -> Option<f64> {
        self.partial.bits(&Unordered(P::from_complete(x.clone())))
    }
}

#[derive(Clone)]
pub struct PartialShuffleCodec<P, D, DFromLen, Pa>
    where P: Partial + GroupPermutable,
          P::Complete: GroupPermutable,
          D: Codec<Symbol=P::Diff>,
          DFromLen: Fn(usize) -> D + Clone,
          Pa: Partitioner<P> {
    pub complete_len: usize,
    pub diff_len: usize,
    pub diff_codec_for_len: DFromLen,
    pub partitioner: Pa,
    pub _marker: PhantomData<P>,
}

impl<P, D, DFromLen, Pa> Codec for PartialShuffleCodec<P, D, DFromLen, Pa> where
    P: Partial + GroupPermutable,
    P::Complete: GroupPermutable,
    D: Codec<Symbol=P::Diff>,
    DFromLen: Fn(usize) -> D + Clone,
    Pa: Partitioner<P> {
    type Symbol = Unordered<P>;

    fn push(&self, m: &mut Message, Unordered(x): &Self::Symbol) {
        assert_eq!(self.len(), x.len());
        if x.len() == 0 { return; }

        let cell = self.cell_codec(&x).pop(m);
        let mut x = &Permutation::swap(*cell.first().unwrap(), x.last()) * x;
        let diff = x.pop();
        self.diff_codec().push(m, &diff);
        self.inner_codec().push(m, &Unordered(x));
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        if self.len() == 0 {
            return Unordered(P::empty(self.complete_len));
        }

        let Unordered(mut x) = self.inner_codec().pop(m);
        let diff = self.diff_codec().pop(m);
        x.push(diff);
        let cell = self.partitioner.cell(&x, x.last());
        self.cell_codec(&x).push(m, &cell);
        Unordered(x)
    }

    fn bits(&self, Unordered(x): &Self::Symbol) -> Option<f64> {
        if self.len() == 0 { return Some(0.); }
        let cell = self.partitioner.cell(x, x.last());
        let orbit_bits = self.cell_codec(x).bits(&cell)?;
        let mut x = x.clone();
        let diff = x.pop();
        Some(self.diff_codec().bits(&diff)? - orbit_bits +
            self.inner_codec().bits(&Unordered(x))?)
    }
}

impl<P, D, DFromLen, Pa> PartialShuffleCodec<P, D, DFromLen, Pa> where
    D: Codec<Symbol=P::Diff>,
    DFromLen: Clone + Fn(usize) -> D,
    P: Partial + GroupPermutable,
    P::Complete: GroupPermutable,
    Pa: Partitioner<P> {
    fn cell_codec(&self, x: &P) -> impl Codec<Symbol=Cell> {
        DistCodec(self.partitioner.cells(x))
    }
}

#[derive(Clone, Debug)]
pub struct OrbitPartitioner;

impl<P: GroupPermutable> Partitioner<P> for OrbitPartitioner {
    type Dist = OrbitDist;

    fn cells(&self, x: &P) -> Self::Dist {
        OrbitDist::new(x.automorphisms())
    }

    fn cell(&self, x: &P, element: usize) -> Cell {
        x.automorphisms().orbit(element)
    }
}

#[derive(Clone, Debug)]
pub struct OrbitDist {
    automorphisms: Automorphisms,
}

impl Distribution for OrbitDist {
    type Symbol = Orbit;

    fn norm(&self) -> usize { self.automorphisms.group.len }
    fn pmf(&self, orbit: &Orbit) -> usize { orbit.len() }

    fn cdf(&self, orbit: &Orbit, i: usize) -> usize {
        *self.canonized(orbit).iter().nth(i).unwrap()
    }

    fn icdf(&self, cf: usize) -> (Orbit, usize) {
        let some_element = &self.automorphisms.decanon * cf;
        let orbit = self.automorphisms.orbit(some_element);
        let i = self.canonized(&orbit).iter().position(|&x| x == cf).unwrap();
        (orbit, i)
    }
}

impl OrbitDist {
    fn new(automorphisms: Automorphisms) -> Self { Self { automorphisms } }

    fn canonized(&self, orbit: &Orbit) -> Orbit {
        Orbit::from_iter(orbit.iter().map(|&x| &self.automorphisms.canon * x))
    }
}

impl<P, D, DFromLen, Pa> PartialShuffleCodec<P, D, DFromLen, Pa> where
    D: Codec<Symbol=P::Diff>,
    DFromLen: Clone + Fn(usize) -> D,
    P: GroupPermutable + Partial,
    P::Complete: GroupPermutable,
    Pa: Partitioner<P> {
    fn len(&self) -> usize {
        self.complete_len - self.diff_len
    }

    fn diff_codec(&self) -> D {
        (self.diff_codec_for_len)(self.diff_len)
    }

    fn inner_codec(&self) -> PartialShuffleCodec<P, D, DFromLen, Pa> {
        PartialShuffleCodec {
            complete_len: self.complete_len,
            diff_len: self.diff_len + 1,
            diff_codec_for_len: self.diff_codec_for_len.clone(),
            partitioner: self.partitioner.clone(),
            _marker: self._marker,
        }
    }
}

impl<P, D, DFromLen, Pa> InterleavedShuffleCodec<P, D, DFromLen, Pa> where
    P: Partial + GroupPermutable,
    P::Complete: GroupPermutable,
    D: Codec<Symbol=P::Diff>,
    DFromLen: Fn(usize) -> D + Clone,
    Pa: Partitioner<P> {
    #[allow(unused)]
    pub fn new(len: usize, diff_codec_for_len: DFromLen, partitioner: Pa) -> Self {
        Self { partial: PartialShuffleCodec { complete_len: len, diff_len: 0, diff_codec_for_len, partitioner, _marker: PhantomData } }
    }
}

#[allow(unused)]
pub fn optimal_interleaved<P, D, DForLen>(len: usize, diff_codec_for_len: DForLen) -> InterleavedShuffleCodec<P, D, DForLen, OrbitPartitioner> where
    P: Partial + GroupPermutable,
    P::Complete: GroupPermutable,
    D: Codec<Symbol=P::Diff>,
    DForLen: Fn(usize) -> D + Clone {
    InterleavedShuffleCodec::new(len, diff_codec_for_len, OrbitPartitioner)
}

pub trait Partitioner<P: Permutable>: Clone {
    type Dist: Distribution<Symbol=Cell>;

    /// Get cells in form of a distribution.
    fn cells(&self, x: &P) -> Self::Dist;
    /// Get cell containing a given `element`.
    fn cell(&self, x: &P, element: usize) -> Cell;
}
