use std::marker::PhantomData;

use crate::codec::Distribution;
use crate::permutable::Orbit;
use crate::plain::Automorphisms;
use crate::recursive::{OrbitCodec, PlainPrefix, UncachedPrefixFn};

/// Orbit codecs for complete recursive shuffle coding using PlainPermutable::automorphisms()
/// at every iteration. Very slow.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlainOrbitCodecs<P: PlainPrefix> {
    phantom: PhantomData<P>,
}

impl<P: PlainPrefix> PlainOrbitCodecs<P> {
    pub fn new() -> Self { Self { phantom: PhantomData } }
}

impl<P: PlainPrefix> UncachedPrefixFn for PlainOrbitCodecs<P> {
    type Prefix = P;
    type Output = PlainOrbitCodec;

    fn apply(&self, x: &P) -> Self::Output {
        PlainOrbitCodec::new(x.automorphisms())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlainOrbitCodec {
    automorphisms: Automorphisms,
}

impl Distribution for PlainOrbitCodec {
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

impl PlainOrbitCodec {
    fn new(automorphisms: Automorphisms) -> Self { Self { automorphisms } }

    fn canonized(&self, orbit: &Orbit) -> Orbit {
        Orbit::from_iter(orbit.iter().map(|&x| &self.automorphisms.canon * x))
    }
}

impl OrbitCodec for PlainOrbitCodec {
    fn id(&self, index: usize) -> Orbit {
        self.automorphisms.orbit(index)
    }

    fn index(&self, id: &Orbit) -> usize { *id.first().unwrap() }
}