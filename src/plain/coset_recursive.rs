//! An interleaved codec to for cosets of permutation groups that can be used with plain shuffle coding. 
//! Reduces initial bits, but not as much as recursive shuffle coding.
use crate::codec::{Codec, Message, MutCategorical, MutDistribution, UniformCodec};
use crate::permutable::Permutation;
use crate::plain::{OrbitElementUniform, OrbitStabilizerInfo, PermutationGroup, PlainPermutable, PlainShuffleCodec, RCosetUniform};

#[derive(Clone, Debug, Eq, PartialEq)]
struct AutomorphicOrbitStabilizer {
    orbit: OrbitElementUniform,
    stabilizer: PermutationGroup,
    element_to_min: Permutation,
}

fn automorphic_orbit_stabilizer(group: PermutationGroup, element: usize) -> AutomorphicOrbitStabilizer {
    let OrbitStabilizerInfo { orbit, stabilizer, from } =
        group.orbit_stabilizer(element);
    let orbit = OrbitElementUniform::new(orbit);
    let min = orbit.min();
    let element_to_min = from[&min].inverse();
    assert_eq!(&element_to_min * element, min);
    let stabilizer = stabilizer.permuted(&element_to_min);
    AutomorphicOrbitStabilizer { orbit, stabilizer, element_to_min }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RecursiveRCosetUniform {
    pub group: PermutationGroup,
    pub stab_codec: MutCategorical,
}

impl Codec for RecursiveRCosetUniform {
    type Symbol = Permutation;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        let mut indices = x.clone();
        let mut stab_codec = self.stab_codec.clone();
        let mut group = self.group.clone();
        let mut codecs = vec![];
        for i in 0..self.group.len {
            let element = &indices.inverse() * i;
            let AutomorphicOrbitStabilizer { orbit, stabilizer, element_to_min } = automorphic_orbit_stabilizer(group, element);
            let min = orbit.min();
            codecs.push((orbit, stab_codec.clone()));
            stab_codec.remove(&min, 1);
            group = stabilizer;
            indices = indices * element_to_min.inverse();
            assert_eq!(&indices * min, i);
        }

        for (orbit, stab) in codecs.iter().rev() {
            let element = orbit.pop(m);
            stab.push(m, &element);
        }
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        let mut group = self.group.clone();
        let mut stab_codec = self.stab_codec.clone();
        let mut elements = vec![];
        for _ in 0..self.group.len {
            let element = stab_codec.pop(m);
            let AutomorphicOrbitStabilizer { orbit, stabilizer, .. } =
                automorphic_orbit_stabilizer(group, element);
            orbit.push(m, &element);
            let min = orbit.min();
            assert!(!elements.contains(&min));
            elements.push(min);
            group = stabilizer;
            stab_codec.remove(&min, 1);
        }

        let indices = Permutation::from(elements).inverse();
        indices
    }

    fn bits(&self, _: &Self::Symbol) -> Option<f64> { Some(self.uni_bits()) }
}

impl UniformCodec for RecursiveRCosetUniform {
    fn uni_bits(&self) -> f64 {
        RCosetUniform::new(self.group.clone()).uni_bits()
    }
}

impl RecursiveRCosetUniform {
    pub fn new(group: PermutationGroup) -> Self {
        let stab_codec = MutCategorical::new((0..group.len).map(|i| (i, 1)));
        Self { group, stab_codec }
    }
}

pub fn coset_recursive_shuffle_codec<C: Codec<Symbol: PlainPermutable>>(codec: C) -> PlainShuffleCodec<C, RecursiveRCosetUniform> {
    PlainShuffleCodec { ordered: codec, rcoset_for: |x| RecursiveRCosetUniform::new(x.automorphism_group()) }
}


#[cfg(test)]
pub mod tests {
    use crate::codec::Codec;
    use crate::graph_codec::tests::small_digraphs;
    use crate::plain::PlainPermutable;

    use super::*;

    #[test]
    fn recursive_rcoset_codec() {
        for x in small_digraphs() {
            RecursiveRCosetUniform::new(x.automorphism_group()).test_on_samples(100);
        }
    }
}