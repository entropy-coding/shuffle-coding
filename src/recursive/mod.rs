//! Recursive shuffle coding.
use std::fmt::Debug;

use crate::codec::{Codec, Message};
use crate::permutable::{Len, Permutable, Unordered};
use crate::plain::PlainPermutable;

pub mod graph;
pub mod multiset;
pub mod prefix_orbit;
pub mod plain_orbit;
pub mod joint;
pub mod product;

/// A prefix of a permutable object, allowing to pop or push a slice to obtain a new prefix.
pub trait Prefix: Permutable {
    type Full: Permutable;
    type Slice;

    fn pop_slice(&mut self) -> Self::Slice;
    fn push_slice(&mut self, slice: &Self::Slice);
    fn from_full(full: Self::Full) -> Self;
    fn into_full(self) -> Self::Full;
    fn last_index(&self) -> usize { self.len() - 1 }
}

/// A function defined on prefixes. If the prefix is the result of a pop or push operation
/// the output can be computed based on the original prefix and the popped/pushed slice, for efficiency.
pub trait PrefixFn: Clone {
    type Prefix: Prefix;
    type Output;

    fn apply(&self, x: &Self::Prefix) -> Self::Output;
    fn update_after_pop_slice(&self, image: &mut Self::Output, x: &Self::Prefix, _slice: &<Self::Prefix as Prefix>::Slice);
    fn update_after_push_slice(&self, image: &mut Self::Output, x: &Self::Prefix, _slice: &<Self::Prefix as Prefix>::Slice);
    fn swap(&self, _image: &mut Self::Output, _i: usize, _j: usize);
}

/// Uncached implementation of PrefixFn, recomputing the output from scratch after each push/pop.
pub trait UncachedPrefixFn: Clone {
    type Prefix: Prefix;
    type Output;
    fn apply(&self, x: &Self::Prefix) -> Self::Output;
}

impl<F: UncachedPrefixFn> PrefixFn for F {
    type Prefix = F::Prefix;
    type Output = F::Output;

    fn apply(&self, x: &Self::Prefix) -> Self::Output {
        self.apply(x)
    }

    fn update_after_pop_slice(&self, image: &mut Self::Output, x: &Self::Prefix, _slice: &<Self::Prefix as Prefix>::Slice) {
        *image = self.apply(x)
    }

    fn update_after_push_slice(&self, image: &mut Self::Output, x: &Self::Prefix, _slice: &<Self::Prefix as Prefix>::Slice) {
        *image = self.apply(x)
    }

    /// The result of swap is only used to update the image passed into update_after_pop/push_slice.
    /// The value of image argument is unused here, so we don't need to implement swap.
    fn swap(&self, _image: &mut Self::Output, _i: usize, _j: usize) {}
}

/// Autoregressive model for permutable objects
/// represented by a function returning a codec for a slice given an adjacent prefix.
pub trait SliceCodecs: PrefixFn<Output: Codec<Symbol=<Self::Prefix as Prefix>::Slice>> + Len {
    fn empty_prefix(&self) -> impl Codec<Symbol=Self::Prefix>;
}

/// A codec for orbits of a permutable object, represented as an orbit id of type Self::Symbol.
pub trait OrbitCodec: Codec {
    /// Returns the orbit id corresponding to the given index from 0..self.len().
    fn id(&self, index: usize) -> Self::Symbol;

    /// Returns an element from the orbit with the given orbit id.
    fn index(&self, id: &Self::Symbol) -> usize;

    /// Push an orbit, given one of its elements.
    fn push_element(&self, m: &mut Message, index: usize) {
        self.push(m, &self.id(index));
    }

    /// Pop an orbit, and return one of its elements.
    fn pop_element(&self, m: &mut Message) -> usize {
        self.index(&self.pop(m))
    }
}

/// A function returning a codec for the orbits of a given prefix.
pub trait OrbitCodecs: PrefixFn<Output: OrbitCodec> {}
impl<O: PrefixFn<Output: OrbitCodec>> OrbitCodecs for O {}

/// We restrict objects and prefixes used for recursive shuffle coding to PlainPermutable so that
/// Unordered<...> implements Eq, which is required for Codec::Symbol.
pub trait PlainPrefix: Prefix<Full: PlainPermutable> + PlainPermutable {}
impl<P: Prefix<Full: PlainPermutable> + PlainPermutable> PlainPrefix for P {}

pub trait PlainSliceCodecs: SliceCodecs<Prefix: PlainPrefix> {}
impl<S: SliceCodecs<Prefix: PlainPrefix>> PlainSliceCodecs for S {}

/// The core of recursive shuffle coding: A recursive codec for prefixes of unordered objects.
/// Written non-recursively for efficiency.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrefixShuffleCodec<S: PlainSliceCodecs, O: OrbitCodecs<Prefix=S::Prefix>> {
    /// Autoregressive model, coding slices given an adjacent prefix.
    pub slice_codecs: S,
    /// Codecs for the orbits of the prefixes of unordered objects.
    pub orbit_codecs: O,
}

impl<S: PlainSliceCodecs, O: OrbitCodecs<Prefix=S::Prefix>> Codec for PrefixShuffleCodec<S, O> {
    type Symbol = Unordered<S::Prefix>;

    fn push(&self, m: &mut Message, Unordered(x): &Self::Symbol) {
        assert_eq!(self.slice_codecs.len(), x.len());
        let x = &mut x.clone();
        let orbit_codec = &mut self.orbit_codecs.apply(x);
        let slice_codec = &mut self.slice_codecs.apply(x);
        for _ in 0..self.slice_codecs.len() {
            let index = orbit_codec.pop_element(m);
            let last_index = x.last_index();
            x.swap(index, last_index);
            self.orbit_codecs.swap(orbit_codec, index, last_index);
            self.slice_codecs.swap(slice_codec, index, last_index);
            let slice = &x.pop_slice();
            self.orbit_codecs.update_after_pop_slice(orbit_codec, x, slice);
            self.slice_codecs.update_after_pop_slice(slice_codec, x, slice);
            slice_codec.push(m, slice);
        }
        self.slice_codecs.empty_prefix().push(m, &x);
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        let mut x = self.slice_codecs.empty_prefix().pop(m);
        let orbit_codec = &mut self.orbit_codecs.apply(&x);
        let slice_codec = &mut self.slice_codecs.apply(&x);
        for _ in 0..self.slice_codecs.len() {
            let slice = &slice_codec.pop(m);
            x.push_slice(slice);
            self.orbit_codecs.update_after_push_slice(orbit_codec, &x, slice);
            self.slice_codecs.update_after_push_slice(slice_codec, &x, slice);
            orbit_codec.push_element(m, x.last_index());
        }
        Unordered(x)
    }

    fn bits(&self, _: &Self::Symbol) -> Option<f64> {
        None
    }
}

/// Implements recursive shuffle coding as a wrapper around PrefixShuffleCodec,
/// mapping prefixes of full length to unordered objects and back.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShuffleCodec<S: PlainSliceCodecs, O: OrbitCodecs<Prefix=S::Prefix>> {
    pub prefix: PrefixShuffleCodec<S, O>,
}

impl<S: PlainSliceCodecs, O: OrbitCodecs<Prefix=S::Prefix>> Codec for ShuffleCodec<S, O> {
    type Symbol = Unordered<<S::Prefix as Prefix>::Full>;

    fn push(&self, m: &mut Message, Unordered(x): &Self::Symbol) {
        self.prefix.push(m, &Unordered(Prefix::from_full(x.clone())));
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        Unordered(self.prefix.pop(m).0.into_full())
    }

    fn bits(&self, Unordered(x): &Self::Symbol) -> Option<f64> {
        self.prefix.bits(&Unordered(Prefix::from_full(x.clone())))
    }
}

impl<S: PlainSliceCodecs, O: OrbitCodecs<Prefix=S::Prefix>> ShuffleCodec<S, O> {
    #[allow(unused)]
    pub fn new(slice_codecs: S, orbit_codecs: O) -> Self {
        Self { prefix: PrefixShuffleCodec { slice_codecs, orbit_codecs } }
    }
}

/// SliceCodecs is not directly implementing codec to avoid blanket implementation conflict 
/// of `impl<S: SliceCodecs> Codec for S` with `impl<D: Distribution> Codec for D`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Autoregressive<S: SliceCodecs>(pub S);

impl<S: SliceCodecs> Codec for Autoregressive<S> {
    type Symbol = <S::Prefix as Prefix>::Full;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        let mut prefix = Prefix::from_full(x.clone());
        let slice_codec = &mut self.0.apply(&prefix);
        for _ in 0..self.0.len() {
            let slice = prefix.pop_slice();
            self.0.update_after_pop_slice(slice_codec, &prefix, &slice);
            slice_codec.push(m, &slice);
        }
        self.0.empty_prefix().push(m, &prefix);
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        let mut prefix = self.0.empty_prefix().pop(m);
        let slice_codec = &mut self.0.apply(&prefix);
        for _ in 0..self.0.len() {
            let slice = slice_codec.pop(m);
            prefix.push_slice(&slice);
            self.0.update_after_push_slice(slice_codec, &prefix, &slice);
        }
        prefix.into_full()
    }

    fn bits(&self, _: &Self::Symbol) -> Option<f64> { None }
}

impl<S: SliceCodecs> Len for Autoregressive<S> {
    fn len(&self) -> usize {
        self.0.len()
    }
}