This is accompanying code for our anonymous paper submission "Entropy Coding of Large Unordered Data Structures".

# Practical Shuffle Coding

Recursive shuffle coding is a general method for optimal compression of unordered objects using bits-back
coding.
Data structures that can be compressed with our method include multisets, graphs, hypergraphs, and others.
Unlike plain shuffle coding, our method allows `one-shot' compression where only a single such object is to be
compressed.

Incomplete shuffle coding allows near-optimal compression of large unordered objects with
intractable automorphism groups.

When combined, these methods achieve state-of-the-art one-shot compression rates on various large network graphs at
competitive speeds.
We release an implementation that can be easily adapted to different data types and statistical models.

## Run

Apart from [Rust](https://www.rust-lang.org/tools/install), you need a C compiler to build this project due to
dependency on [nauty and Traces](https://github.com/a-maier/nauty-Traces-sys).

The following commands replicate all experiments:

```shell
cargo run --release -- --stats --joint --er --pu --wl-iter=1 SZIP
cargo run --release -- --stats --recursive --ae --ap --wl-iter=1 SZIP
cargo run --release -- --stats --recursive --ap --wl-iter=2 SZIP
cargo run --release -- --stats --recursive --ap --wl-iter=0 REC
cargo run --release -- --stats --recursive --ap --wl-iter=1 Gowalla DBLP
cargo test --release benchmark_multiset -- --ignored
```

Graph datasets are downloaded automatically as
needed ([TU](https://chrsmrrs.github.io/datasets/), [SZIP](https://github.com/juliuskunze/szip-graphs)
and [REC](https://github.com/entropy-coding/rec-graphs)). Full results can be
found [here](https://docs.google.com/spreadsheets/d/1YP0om6hNktaUhyFzFVVSh5SCTefZ8kNafYDYYIDolhM/edit#gid=1483743764).

Use `cargo run -- --help` to see all available options.

## Documentation

The following files implement and explain the main concepts.
Reading them in this order will give you a good overview:

1. [`ans.rs`](src/ans.rs): Asymmetric numeral systems (ANS).
2. [`permutable.rs`](src/permutable.rs): Ordered/unordered objects.
3. [`recursive/mod.rs`](src/recursive/mod.rs): Recursive shuffle coding.

## Module structure

- [`plain`](src/plain): Plain shuffle coding.
- [`recursive`](src/recursive): Recursive shuffle coding, in particular [`joint.rs`](src/recursive/joint.rs) for joint
  shuffle coding, and with multisets
  in [`multiset.rs`](src/recursive/multiset.rs).
- [`recursive/graph`](src/recursive/graph): Recursive shuffle coding on graphs, in
  particular [`incomplete.rs`](src/recursive/graph/incomplete.rs) for incomplete shuffle coding.

## Credits

This code is based off
the [original implementation of the paper "Entropy Coding of Unordered Data Structures"](https://github.com/juliuskunze/shuffle-coding).
