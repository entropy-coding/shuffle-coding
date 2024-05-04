//! Joint shuffle coding.
use crate::codec::{Codec, ConstantCodec, Message};
use crate::permutable::{Permutable, PermutableCodec};
use crate::recursive::{Len, Prefix, SliceCodecs, UncachedPrefixFn};

/// Prefix type forming the basis for joint shuffle coding, where the empty (length 0) prefix
/// contains the full permutable object, and all slices are empty.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JointPrefix<P: Permutable> {
    pub full: P,
    pub len: usize,
}

impl<P: Permutable> Prefix for JointPrefix<P> {
    type Full = P;
    type Slice = ();

    fn pop_slice(&mut self) {
        self.len -= 1;
    }

    fn push_slice(&mut self, _: &()) {
        self.len += 1;
    }

    fn from_full(full: Self::Full) -> Self {
        Self { len: full.len(), full }
    }

    fn into_full(self) -> Self::Full {
        self.full
    }
}

impl<P: Permutable> Permutable for JointPrefix<P> {
    fn len(&self) -> usize {
        self.len
    }

    fn swap(&mut self, i: usize, j: usize) {
        assert!(i < self.len());
        assert!(j < self.len());
        self.full.swap(i, j)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JointSliceCodecs<C: PermutableCodec> {
    pub empty: EmptyJointPrefixCodec<C>,
}

impl<C: PermutableCodec> UncachedPrefixFn for JointSliceCodecs<C> {
    type Prefix = JointPrefix<C::Symbol>;
    type Output = ConstantCodec<()>;

    fn apply(&self, _: &Self::Prefix) -> Self::Output {
        ConstantCodec(())
    }
}

impl<C: PermutableCodec> Len for JointSliceCodecs<C> {
    fn len(&self) -> usize {
        self.empty.full.len()
    }
}

impl<C: PermutableCodec> SliceCodecs for JointSliceCodecs<C> {
    fn empty_prefix(&self) -> impl Codec<Symbol=Self::Prefix> {
        self.empty.clone()
    }
}

impl<C: PermutableCodec> JointSliceCodecs<C> {
    pub fn new(full: C) -> Self {
        Self { empty: EmptyJointPrefixCodec { full } }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EmptyJointPrefixCodec<C: PermutableCodec> {
    pub full: C,
}

impl<C: PermutableCodec> Codec for EmptyJointPrefixCodec<C> {
    type Symbol = JointPrefix<C::Symbol>;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        self.full.push(m, &x.full)
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        JointPrefix { len: 0, full: self.full.pop(m) }
    }

    fn bits(&self, x: &Self::Symbol) -> Option<f64> {
        self.full.bits(&x.full)
    }
}