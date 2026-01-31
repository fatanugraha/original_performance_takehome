"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        # Simple slot packing that just uses one slot per instruction bundle
        instrs = []
        for engine, slot in slots:
            instrs.append({engine: [slot]})
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_hash(self, val_hash_addr, tmp1, tmp2, t_hash_magic, t_hash_magic_2, round, i):
        self.instrs.append({"valu": [
            ("+", tmp1, val_hash_addr, t_hash_magic+0*VLEN),
            ("<<", tmp2, val_hash_addr, t_hash_magic_2+0*VLEN),
        ]})
        self.instrs.append({"valu": [("+", val_hash_addr, tmp1, tmp2)]})

        self.instrs.append({"valu": [
            ("^", tmp1, val_hash_addr, t_hash_magic+1*VLEN),
            (">>", tmp2, val_hash_addr, t_hash_magic_2+1*VLEN),
        ]})
        self.instrs.append({"valu": [("^", val_hash_addr, tmp1, tmp2)]})

        self.instrs.append({"valu": [
            ("+", tmp1, val_hash_addr, t_hash_magic+2*VLEN),
            ("<<", tmp2, val_hash_addr, t_hash_magic_2+2*VLEN),
        ]})
        self.instrs.append({"valu": [("+", val_hash_addr, tmp1, tmp2)]})

        self.instrs.append({"valu": [
            ("+", tmp1, val_hash_addr, t_hash_magic+3*VLEN),
            ("<<", tmp2, val_hash_addr, t_hash_magic_2+3*VLEN),
        ]})
        self.instrs.append({"valu": [("^", val_hash_addr, tmp1, tmp2)]})

        self.instrs.append({"valu": [
            ("+", tmp1, val_hash_addr, t_hash_magic+4*VLEN),
            ("<<", tmp2, val_hash_addr, t_hash_magic_2+4*VLEN),
        ]})
        self.instrs.append({"valu": [("+", val_hash_addr, tmp1, tmp2)]})

        self.instrs.append({"valu": [
            ("^", tmp1, val_hash_addr, t_hash_magic+5*VLEN),
            (">>", tmp2, val_hash_addr, t_hash_magic_2+5*VLEN),
        ]})
        self.instrs.append({"valu": [("^", val_hash_addr, tmp1, tmp2)]})

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Like reference_kernel2 but building actual instructions.
        Scalar implementation using only scalar ALU and load/store.
        """
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp3 = self.alloc_scratch("tmp3")

        # Scratch space addresses
        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)

        # Pause instructions are matched up with yield statements in the reference
        # kernel to let you debug at intermediate steps. The testing harness in this
        # file requires these match up to the reference kernel's yields, but the
        # submission harness ignores them.
        self.add("flow", ("pause",))
        # Any debug engine instruction is ignored by the submission simulator
        self.add("debug", ("comment", "Starting loop"))

        val_base_addr = SCRATCH_SIZE-batch_size
        idx_base_addr = val_base_addr-batch_size
        t_node_val_p = idx_base_addr - VLEN
        t_branch_2x = t_node_val_p - VLEN
        t_node_val = t_branch_2x - VLEN
        t_hash_val = t_node_val - VLEN
        t_left = t_hash_val - VLEN
        t_right = t_left - VLEN
        t_twos = t_right - VLEN
        t_ones = t_twos - VLEN
        t_n_nodes = t_ones - VLEN
        t_zeros = t_n_nodes - VLEN
        t_tmp1 = t_zeros - VLEN
        t_tmp2 = t_tmp1 - VLEN
        t_hash_magic = t_tmp2 - (6 * VLEN)
        t_hash_magic_2 = t_hash_magic - (6 * VLEN)

        # load idx and vals to scratch
        self.instrs.append({
            "alu": [("+", tmp3, zero_const, self.scratch['inp_values_p'])],
            "load": [("const", tmp1, VLEN)],
            "valu": [
                ("vbroadcast", t_zeros, zero_const), # gp1 = 2*idx [t]
                ("vbroadcast", t_twos, two_const), # gp1 = 2*idx [t]
                ("vbroadcast", t_ones, one_const), # gp1 = 2*idx [t]
                ("vbroadcast", t_n_nodes, self.scratch["n_nodes"]), # gp1 = 2*idx [t]
            ],
        })
        self.instrs.append({
            "valu": [
                ("vbroadcast", t_hash_magic+0*VLEN, self.scratch_const(0x7ED55D16)),
                ("vbroadcast", t_hash_magic+1*VLEN, self.scratch_const(0xC761C23C)),
                ("vbroadcast", t_hash_magic+2*VLEN, self.scratch_const(0x165667B1)),
                ("vbroadcast", t_hash_magic+3*VLEN, self.scratch_const(0xD3A2646C)),
                ("vbroadcast", t_hash_magic+4*VLEN, self.scratch_const(0xFD7046C5)),
                ("vbroadcast", t_hash_magic+5*VLEN, self.scratch_const(0xB55A4F09)),
            ]
        })
        self.instrs.append({
            "valu": [
                ("vbroadcast", t_hash_magic_2+0*VLEN, self.scratch_const(12)),
                ("vbroadcast", t_hash_magic_2+1*VLEN, self.scratch_const(19)),
                ("vbroadcast", t_hash_magic_2+2*VLEN, self.scratch_const(5)),
                ("vbroadcast", t_hash_magic_2+3*VLEN, self.scratch_const(9)),
                ("vbroadcast", t_hash_magic_2+4*VLEN, self.scratch_const(3)),
                ("vbroadcast", t_hash_magic_2+5*VLEN, self.scratch_const(16)),
            ]
        })
        for i in range(0, 256, 8):
            self.instrs.append({
                "load": [("vload", val_base_addr+i, tmp3)],
                "alu": [("+", tmp3, tmp3, tmp1)],
            })

        for round in range(rounds):
            for gid in range(0, batch_size, 8):
                gidx_addr = idx_base_addr + gid
                gval_addr = val_base_addr + gid

                self.instrs.append({"valu": [
                    ("vbroadcast", t_node_val_p, self.scratch["forest_values_p"]), # gp0 = node_val_p [t]
                ]})

                self.instrs.append({"valu": [
                    ("+", t_node_val_p, t_node_val_p, gidx_addr),
                    ("*", t_branch_2x, t_twos, gidx_addr)
                ]})

                for tid in range(8):
                    self.instrs.append({"load": [
                        ("load", t_node_val+tid, t_node_val_p+tid) # gp2 = node_val [t], gp0 is free to use now
                    ]})

                for tid in range(0, VLEN):
                    i = gid+tid
                    self.instrs.append({"debug": [("compare", gidx_addr+tid, (round, i, "idx"))]})
                    self.instrs.append({"debug": [("compare", gval_addr+tid, (round, i, "val"))]})
                    self.instrs.append({"debug": [("compare", t_node_val+tid, (round, i, "node_val"))]})

                self.instrs.append({"valu": [
                    ("^", gval_addr, gval_addr, t_node_val),
                    ("+", t_left, t_branch_2x, t_ones), # speculate next branch.
                    ("+", t_right, t_branch_2x, t_twos), # speculate next branch.
                ]})

                self.build_hash(gval_addr, t_tmp1, t_tmp2, t_hash_magic, t_hash_magic_2, round, i)

                for tid in range(0, VLEN):
                    i = gid+tid
                    self.instrs.append({"debug": [("compare", gval_addr+tid, (round, i, "hashed_val"))]})

                # idx = 2*idx + (1 if val % 2 == 0 else 2)
                self.instrs.append({"valu": [("%", t_hash_val, gval_addr, t_twos)]})
                self.instrs.append({"flow": [("vselect", gidx_addr, t_hash_val, t_right, t_left)]})

                for tid in range(0, VLEN):
                    i = gid+tid
                    self.instrs.append({"debug": [("compare", gidx_addr+tid, (round, i, "next_idx"))]})

                # # # idx = 0 if idx >= n_nodes else idx
                self.instrs.append({"valu": [("<", t_hash_val, gidx_addr, t_n_nodes)]})
                self.instrs.append({"flow": [("vselect", gidx_addr, t_hash_val, gidx_addr, t_zeros)]})
                for tid in range(0, VLEN):
                    i = gid+tid
                    self.instrs.append({"debug": [("compare", gidx_addr+tid, (round, i, "wrapped_idx"))]})

        # write back the idx and vals from scratch to mem
        self.instrs.append({
            "load": [("const", tmp1, VLEN)],
            "alu": [
                ("+", tmp2, zero_const, self.scratch['inp_indices_p']),
                ("+", tmp3, zero_const, self.scratch['inp_values_p'])
            ],
        })
        for i in range(0, 256, 8):
            self.instrs.append({
                "store": [
                    ("vstore", tmp2, idx_base_addr+i),
                    ("vstore", tmp3, val_base_addr+i),
                ],
                "alu": [("+", tmp3, tmp3, tmp1), ("+", tmp2, tmp2, tmp1)],
            })

        # Required to match with the yield in reference_kernel2
        self.instrs.append({"flow": [("pause",)]})


BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()

        # print(machine.mem)
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
