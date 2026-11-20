// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "common/device_fixture.hpp"

#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "impl/program/program_impl.hpp"
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_trace_id.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/workload/specification/program_spec.hpp>
#include <tt-metalium/tensor/mesh_tensor.hpp>
#include <tt-metalium/tensor/tensor_apis.hpp>
#include "test_gold_impls.hpp"
#include "impl/data_format/bfloat16_utils.hpp"
#include "tt_metal/impl/dispatch/slow_dispatch.hpp"

using std::vector;
using namespace tt;
using namespace tt::tt_metal;

namespace {

const DFBSpecName SRC0_DFB{"src0_dfb"};
const DFBSpecName SRC1_DFB{"src1_dfb"};
const DFBSpecName DST_DFB{"dst_dfb"};
const TensorParamName SRC0_T{"src0"};
const TensorParamName SRC1_T{"src1"};
const TensorParamName DST_T{"dst"};
const KernelSpecName READER{"reader"};
const KernelSpecName WRITER{"writer"};
const KernelSpecName COMPUTE{"compute"};

struct BmmParams {
    std::uint32_t Mt, Kt, Nt;
    std::uint32_t B_total;     // total batch count (buffer sizing + validation)
    std::uint32_t B_per_core;  // batch count per core (kernel runtime args)
    std::uint32_t num_threads = 2;     // compute threads (Tensix NEOs on Quasar)
    std::uint32_t reader_threads = 0;  // 0: num_threads
    std::uint32_t writer_threads = 0;  // 0: num_threads
    std::uint32_t num_input_tiles = 4;
    std::uint32_t num_input_tiles_B = 0;  // src1 entries; 0: num_input_tiles
    bool posted_writes = false;           // writer_bmm_8bank_posted.cpp: C goes to DRAM as posted writes
    std::uint32_t num_output_tiles = 4;
    std::uint32_t single_tile_size = 2 * 1024;
    std::uint32_t transpose = 0;
};

struct BmmTensors {
    MeshTensor src0;
    MeshTensor src1;
    MeshTensor dst;
};

// Flat 2D UINT32 page layout: one DRAM page per tile, tile_size bytes each. The element type
// is UINT32 only so DRAM exposes raw tile-paged storage at the buffer level; the kernels still
// operate on bfloat16 tiles via TensorAccessor / matmul LLKs.
TensorSpec make_flat_dram_tensor_spec(std::uint32_t tile_size, std::uint32_t num_tiles) {
    const std::uint32_t tile_size_words = tile_size / sizeof(std::uint32_t);
    auto page_config = PageConfig(Layout::ROW_MAJOR);
    auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    auto tensor_layout = TensorLayout(DataType::UINT32, page_config, memory_config);
    return TensorSpec(Shape{num_tiles, tile_size_words}, tensor_layout);
}

BmmTensors create_bmm_tensors(distributed::MeshDevice& mesh_device, const BmmParams& p) {
    const std::uint32_t num_tiles_A = p.Mt * p.Kt * p.B_total;
    const std::uint32_t num_tiles_B = p.Kt * p.Nt * p.B_total;
    const std::uint32_t num_tiles_C = p.Mt * p.Nt * p.B_total;
    return {
        MeshTensor::allocate_on_device(mesh_device, make_flat_dram_tensor_spec(p.single_tile_size, num_tiles_A)),
        MeshTensor::allocate_on_device(mesh_device, make_flat_dram_tensor_spec(p.single_tile_size, num_tiles_B)),
        MeshTensor::allocate_on_device(mesh_device, make_flat_dram_tensor_spec(p.single_tile_size, num_tiles_C)),
    };
}

template <typename TargetNodes>
ProgramSpec build_bmm_program_spec(
    const BmmParams& p, const BmmTensors& tensors, const TargetNodes& target_nodes, bool use_implicit_sync) {
    // Data movement configs are chosen per arch so the same spec runs on WH/BH and
    // Quasar. The kernel sources use the unified
    // DataflowBuffer device API on both arches (CB-backed on Gen1, DFB-backed on Gen2).
    // On Quasar we also enable implicit sync on each DFB so the reader/writer kernels can drop
    // explicit reserve_back/wait_front/push_back/pop_front; on WH/BH implicit sync is unsupported
    // and must be disabled to match the explicit-sync kernel branch.
    const bool is_quasar = tensors.src0.device().arch() == ARCH::QUASAR;
    DataMovementHardwareConfig reader_config;
    DataMovementHardwareConfig writer_config;
    ComputeHardwareConfig compute_config;
    if (is_quasar) {
        reader_config = DataMovementHardwareConfig{
            .config_2xx =
                DataMovementHardwareConfig::DataMovement2XXConfig{
                    .disable_dfb_implicit_sync_for_all = !use_implicit_sync,
                },
        };
        writer_config = DataMovementHardwareConfig{
            .config_2xx =
                DataMovementHardwareConfig::DataMovement2XXConfig{
                    .disable_dfb_implicit_sync_for_all = !use_implicit_sync,
                },
        };
        compute_config = ComputeHardwareConfig{};
    } else {
        reader_config = DataMovementHardwareConfig{
            .config_1xx =
                DataMovementHardwareConfig::DataMovement1XXConfig{
                    .processor = DataMovementProcessor::RISCV_1,
                    .noc = NOC::RISCV_1_default,
                },
        };
        writer_config = DataMovementHardwareConfig{
            .config_1xx =
                DataMovementHardwareConfig::DataMovement1XXConfig{
                    .processor = DataMovementProcessor::RISCV_0,
                    .noc = NOC::RISCV_0_default,
                },
        };
        compute_config = ComputeHardwareConfig{};
    }

    DataflowBufferSpec src0_dfb_spec{
        .unique_id = SRC0_DFB,
        .entry_size = p.single_tile_size,
        .num_entries = p.num_input_tiles,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };
    DataflowBufferSpec src1_dfb_spec{
        .unique_id = SRC1_DFB,
        .entry_size = p.single_tile_size,
        .num_entries = p.num_input_tiles_B ? p.num_input_tiles_B : p.num_input_tiles,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };
    DataflowBufferSpec dst_dfb_spec{
        .unique_id = DST_DFB,
        .entry_size = p.single_tile_size,
        .num_entries = p.num_output_tiles,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };

    KernelSpec reader_spec{
        .unique_id = READER,
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_bmm_8bank.cpp",
        .num_threads = p.reader_threads ? p.reader_threads : p.num_threads,
        .dfb_bindings = {ProducerOf(SRC0_DFB, "src0"), ProducerOf(SRC1_DFB, "src1")},
        .tensor_bindings =
            {{.tensor_parameter_name = SRC0_T, .accessor_name = "src0"},
             {.tensor_parameter_name = SRC1_T, .accessor_name = "src1"}},
        // Only batch_start varies per node; everything else is identical across nodes and
        // lives in CRTAs for better dispatch efficiency.
        .runtime_arg_schema =
            {.runtime_arg_names = {"batch_start"},
             .common_runtime_arg_names = {"Mt", "Kt", "Nt", "MtKt", "KtNt", "batch", "do_bcast"}},
        .hw_config = reader_config,
    };

    KernelSpec writer_spec{
        .unique_id = WRITER,
        .source = p.posted_writes ? "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_bmm_8bank_posted.cpp"
                                  : "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_bmm_8bank.cpp",
        .num_threads = p.writer_threads ? p.writer_threads : p.num_threads,
        .dfb_bindings = {ConsumerOf(DST_DFB, "dst")},
        .tensor_bindings = {{.tensor_parameter_name = DST_T, .accessor_name = "dst"}},
        .runtime_arg_schema = {.runtime_arg_names = {"batch_start"}, .common_runtime_arg_names = {"Mt", "Nt", "batch"}},
        .hw_config = writer_config,
    };

    KernelSpec compute_spec{
        .unique_id = COMPUTE,
        .source = "tests/tt_metal/tt_metal/test_kernels/compute/bmm.cpp",
        .num_threads = p.num_threads,
        .dfb_bindings = {ConsumerOf(SRC0_DFB, "src0"), AllConsumerOf(SRC1_DFB, "src1"), ProducerOf(DST_DFB, "dst")},
        .compile_time_args =
            {{"batch", p.B_per_core}, {"Mt", p.Mt}, {"Kt", p.Kt}, {"Nt", p.Nt}, {"transpose", p.transpose}},
        .hw_config = compute_config,
    };

    return ProgramSpec{
        .name = "bmm",
        .kernels = {reader_spec, writer_spec, compute_spec},
        .dataflow_buffers = {src0_dfb_spec, src1_dfb_spec, dst_dfb_spec},
        .tensor_parameters =
            {{.unique_id = SRC0_T, .spec = tensors.src0.tensor_spec()},
             {.unique_id = SRC1_T, .spec = tensors.src1.tensor_spec()},
             {.unique_id = DST_T, .spec = tensors.dst.tensor_spec()}},
        .work_units = {{.name = "main", .kernels = {READER, WRITER, COMPUTE}, .target_nodes = target_nodes}},
    };
}

bool validate_bmm_result(
    const BmmParams& p,
    const std::vector<std::uint32_t>& src0_vec,
    const std::vector<std::uint32_t>& src1_vec,
    const std::vector<std::uint32_t>& result_vec,
    int* argfail) {
    auto comparison_function = [](float a, float b) {
        const float rtol = 0.05f;
        const float atol = 0.05f;
        float maxabs = fmaxf(fabsf(a), fabsf(b));
        float absdiff = fabsf(a - b);
        return (absdiff <= atol) || absdiff < rtol * maxabs;
    };
    vector<std::uint32_t> shapeA = {1, p.B_total, p.Mt * 32, p.Kt * 32};
    vector<std::uint32_t> shapeB = {1, p.B_total, p.Kt * 32, p.Nt * 32};
    vector<std::uint32_t> shapeC = {1, p.B_total, p.Mt * 32, p.Nt * 32};
    auto u16_src0 = u16_from_u32_vector(src0_vec);
    auto u16_src1 = u16_from_u32_vector(src1_vec);
    auto src0_linear = convert_layout<std::uint16_t>(
        u16_src0, shapeA, TensorLayoutType::TILED_NFACES, TensorLayoutType::LIN_ROW_MAJOR);

    const bool transpose_tiles = (p.transpose != 0);

    auto src1_linear = convert_layout<std::uint16_t>(
        u16_src1,
        shapeB,
        TensorLayoutType::TILED_NFACES,
        TensorLayoutType::LIN_ROW_MAJOR,
        std::nullopt,
        std::nullopt,
        transpose_tiles,
        transpose_tiles);

    auto ref_bmm = gold_bmm(shapeA, src0_linear, shapeB, src1_linear);
    auto gold = u32_from_u16_vector(convert_layout<std::uint16_t>(
        ref_bmm, shapeC, TensorLayoutType::LIN_ROW_MAJOR, TensorLayoutType::TILED_NFACES));
    return packed_uint32_t_vector_comparison(result_vec, gold, comparison_function, argfail);
}

void set_bmm_run_args(Program& program, const BmmParams& p, const BmmTensors& tensors, const NodeCoord& node) {
    constexpr std::uint32_t do_bcast = 0;
    ProgramRunArgs params;
    params.kernel_run_args = {
        ProgramRunArgs::KernelRunArgs{
            .kernel = READER,
            .runtime_arg_values = MakeRuntimeArgsForSingleNode(node, {{"batch_start", 0u}}),
            .common_runtime_arg_values =
                {{"Mt", p.Mt},
                 {"Kt", p.Kt},
                 {"Nt", p.Nt},
                 {"MtKt", p.Mt * p.Kt},
                 {"KtNt", p.Kt * p.Nt},
                 {"batch", p.B_per_core},
                 {"do_bcast", do_bcast}},
        },
        ProgramRunArgs::KernelRunArgs{
            .kernel = WRITER,
            .runtime_arg_values = MakeRuntimeArgsForSingleNode(node, {{"batch_start", 0u}}),
            .common_runtime_arg_values = {{"Mt", p.Mt}, {"Nt", p.Nt}, {"batch", p.B_per_core}},
        },
        ProgramRunArgs::KernelRunArgs{.kernel = COMPUTE},
    };
    params.tensor_args = {
        {SRC0_T, TensorArgument{tensors.src0}},
        {SRC1_T, TensorArgument{tensors.src1}},
        {DST_T, TensorArgument{tensors.dst}},
    };
    program.set_program_run_args(params);
}

struct BmmInputs {
    std::vector<std::uint32_t> src0;
    std::vector<std::uint32_t> src1;
};

BmmInputs make_bmm_inputs(const BmmParams& p) {
    const std::uint32_t bytesA = p.single_tile_size * p.Mt * p.Kt * p.B_total;
    const std::uint32_t bytesB = p.single_tile_size * p.Kt * p.Nt * p.B_total;
    return BmmInputs{
        .src0 = create_random_vector_of_bfloat16(bytesA, 1.0f, 0x1234),
        .src1 = create_random_vector_of_bfloat16(bytesB, 1.0f, 0x1234, -0.45f),
    };
}

BmmParams single_node_bmm_params(ARCH arch) {
    BmmParams p;
    if (arch != ARCH::QUASAR) {
        p.Mt = 4;
        p.Kt = 2;
        p.Nt = 3;
        p.B_total = 2;
        p.B_per_core = 2;
        p.num_input_tiles = 2;
        p.num_output_tiles = 2;
        p.num_threads = 1;
    } else {
        p.Mt = 2;
        p.Kt = 2;
        p.Nt = 2;
        p.B_total = 1;
        p.B_per_core = 1;
        p.num_threads = 2;
    }
    return p;
}

void check_bmm_output(
    distributed::MeshCommandQueue& cq, const BmmParams& p, const BmmTensors& tensors, const BmmInputs& inputs) {
    auto result_vec = cq.enqueue_read_tensor(tensors.dst).to_vector<std::uint32_t>();

    int argfail = -1;
    bool pass = validate_bmm_result(p, inputs.src0, inputs.src1, result_vec, &argfail);
    EXPECT_TRUE(pass) << "Failure position=" << argfail;
}

struct BmmSingleNode {
    distributed::MeshWorkload workload;
    BmmInputs inputs;
};

// Builds the single-node bmm workload and writes its inputs. The workload binds `tensors` by
// reference, so the caller keeps them alive.
BmmSingleNode setup_bmm_single_node(distributed::MeshDevice& mesh_device, const BmmParams& p, BmmTensors& tensors) {
    const NodeCoord node{0, 0};
    const bool use_implicit_sync = (mesh_device.arch() == ARCH::QUASAR);
    auto spec = build_bmm_program_spec(p, tensors, node, use_implicit_sync);
    BmmSingleNode bmm{
        .workload = tt_metal::distributed::MeshWorkload(mesh_device, spec),
        .inputs = make_bmm_inputs(p),
    };
    set_bmm_run_args(bmm.workload.get_programs().begin()->second, p, tensors, node);

    auto& cq = mesh_device.mesh_command_queue();
    cq.enqueue_write_tensor(HostTensor::from_vector(bmm.inputs.src0, tensors.src0.tensor_spec()), tensors.src0);
    cq.enqueue_write_tensor(HostTensor::from_vector(bmm.inputs.src1, tensors.src1.tensor_spec()), tensors.src1);
    return bmm;
}

void run_bmm_single_node(distributed::MeshDevice& mesh_device, const BmmParams& p) {
    BmmTensors tensors = create_bmm_tensors(mesh_device, p);
    BmmSingleNode bmm = setup_bmm_single_node(mesh_device, p, tensors);
    auto& cq = mesh_device.mesh_command_queue();
    distributed::EnqueueMeshWorkload(cq, bmm.workload, /*blocking=*/true);
    check_bmm_output(cq, p, tensors, bmm.inputs);
}
}  // namespace

TEST_F(AnyDispatchMeshDeviceSingleCardFixture, Bmm) {
    auto& mesh_device = *devices_[0];
    run_bmm_single_node(mesh_device, single_node_bmm_params(mesh_device.arch()));
}

// Every Tensix NEO of a Quasar node, fed from DRAM. Each kernel splits the output rows by thread
// (row mt belongs to thread mt % threads): 4 readers fetch A's row mt and, with Kt = 4, B's column
// (the reader whose id is both mt and kt fetches B(kt, nt), so B's column comes once, in kt order,
// and every NEO sees it); 4 NEOs compute a row each; 2 writers take rows mt % 2, which with Nt = 1
// is exactly the STRIDED 4 -> 2 dealing of the output buffer. 4 + 2 DMs fit the 6 user DMs.
TEST_F(QuasarMeshDeviceSingleCardFixture, BmmAllNeos) {
    auto& mesh_device = *devices_[0];
    BmmParams p;
    p.Mt = 4;
    p.Kt = 4;
    p.Nt = 1;
    p.B_total = 1;
    p.B_per_core = 1;
    p.num_threads = 4;
    p.reader_threads = 4;
    p.writer_threads = 2;
    // Implicit sync credits a DFB only when every producer has done its share of a txn-id batch, so B(0)
    // reaches the NEOs only once all 4 readers have issued their B tile. Reader r issues B(r) after A(r,0..r),
    // so each reader's share of src0 must hold 4 tiles (16 / 4 producers), or the readers block on A while
    // the NEOs wait on B. With 8 entries (2 per producer) this deadlocks.
    p.num_input_tiles = 16;
    p.num_output_tiles = 8;
    run_bmm_single_node(mesh_device, p);
}

TEST_F(AnyDispatchMeshDeviceSingleCardFixture, BmmTranspose) {
    auto& mesh_device = *devices_[0];
    BmmParams p = single_node_bmm_params(mesh_device.arch());
    p.transpose = 1;
    run_bmm_single_node(mesh_device, p);
}

// This needs to be a separate test because we don't have a way of querying the correct compute grid size
// when running a multi-neo emu/sim build. Otherwise its the same test with batch split across nodes.
TEST_F(QuasarMeshDeviceSingleCardFixture, BmmMultinode) {
    auto& mesh_device = *devices_[0];
    if (mesh_device.compute_with_storage_grid_size().x < 2) {
        GTEST_SKIP() << "This test requires at least 2 worker nodes.";
    }

    BmmParams p;
    p.Mt = 2;
    p.Kt = 2;
    p.Nt = 2;
    p.B_total = 2;     // total batches across both cores
    p.B_per_core = 1;  // each core computes exactly one batch
    p.num_threads = 2;

    auto tensors = create_bmm_tensors(mesh_device, p);

    const NodeCoord node0{0, 0};
    const NodeCoord node1{1, 0};
    const NodeRange node_range{node0, node1};

    // QuasarMeshDeviceSingleCardFixture only opens Quasar devices, so implicit sync is always on.
    auto spec = build_bmm_program_spec(p, tensors, node_range, /*use_implicit_sync=*/true);
    tt_metal::Program program(mesh_device, spec);

    constexpr std::uint32_t do_bcast = 0;
    // node0 handles batch 0, node1 handles batch 1 (batch_start = node index)
    ProgramRunArgs params;
    params.kernel_run_args = {
        ProgramRunArgs::KernelRunArgs{
            .kernel = READER,
            .runtime_arg_values = {{"batch_start", {{node0, 0u}, {node1, 1u}}}},
            .common_runtime_arg_values =
                {{"Mt", p.Mt},
                 {"Kt", p.Kt},
                 {"Nt", p.Nt},
                 {"MtKt", p.Mt * p.Kt},
                 {"KtNt", p.Kt * p.Nt},
                 {"batch", p.B_per_core},
                 {"do_bcast", do_bcast}},
        },
        ProgramRunArgs::KernelRunArgs{
            .kernel = WRITER,
            .runtime_arg_values = {{"batch_start", {{node0, 0u}, {node1, 1u}}}},
            .common_runtime_arg_values = {{"Mt", p.Mt}, {"Nt", p.Nt}, {"batch", p.B_per_core}},
        },
        ProgramRunArgs::KernelRunArgs{.kernel = COMPUTE},
    };
    params.tensor_args = {
        {SRC0_T, TensorArgument{tensors.src0}},
        {SRC1_T, TensorArgument{tensors.src1}},
        {DST_T, TensorArgument{tensors.dst}},
    };
    program.set_program_run_args(params);

    const BmmInputs inputs = make_bmm_inputs(p);
    // MeshTensor doesn't yet expose slow-dispatch read/write APIs; use unit-mesh MeshBuffer helpers.
    slow_dispatch::WriteToBuffer(tensors.src0.mesh_buffer(), inputs.src0);
    slow_dispatch::WriteToBuffer(tensors.src1.mesh_buffer(), inputs.src1);

    LaunchProgram(mesh_device, std::move(program));

    std::vector<std::uint32_t> result_vec;
    slow_dispatch::ReadFromBuffer(tensors.dst.mesh_buffer(), result_vec);

    int argfail = -1;
    bool pass = validate_bmm_result(p, inputs.src0, inputs.src1, result_vec, &argfail);
    EXPECT_TRUE(pass) << "Failure position=" << argfail;
}

// BmmAllNeos's layout (4 readers, 4 NEOs, 2 writers) on every worker node of the device at once: each node
// computes its own B_per_core batches (batch_start = node index * B_per_core), so every Tensix NEO of the grid
// is busy and all nodes move their tiles to and from DRAM concurrently. The program is launched `launches`
// times; dst is zeroed before each launch so a launch cannot pass on its predecessor's output.
void run_bmm_all_neos_all_nodes(distributed::MeshDevice& mesh_device, BmmParams p, std::uint32_t launches) {
    const auto grid = mesh_device.compute_with_storage_grid_size();
    const std::uint32_t num_nodes = grid.x * grid.y;
    if (num_nodes < 2) {
        GTEST_SKIP() << "This test requires at least 2 worker nodes.";
    }
    p.B_total = num_nodes * p.B_per_core;
    p.num_threads = 4;
    p.reader_threads = 4;
    p.writer_threads = 2;

    auto tensors = create_bmm_tensors(mesh_device, p);
    const NodeRange node_range{{0, 0}, {grid.x - 1, grid.y - 1}};

    ProgramRunArgs::KernelRunArgs::RuntimeArgValues batch_start;
    std::uint32_t index = 0;
    for (std::uint32_t y = 0; y < grid.y; y++) {
        for (std::uint32_t x = 0; x < grid.x; x++) {
            AddRuntimeArgsForNode(
                batch_start, NodeCoord{x, y}, {{"batch_start", index++ * p.B_per_core}});
        }
    }

    const BmmInputs inputs = make_bmm_inputs(p);
    slow_dispatch::WriteToBuffer(tensors.src0.mesh_buffer(), inputs.src0);
    slow_dispatch::WriteToBuffer(tensors.src1.mesh_buffer(), inputs.src1);
    const std::vector<std::uint32_t> zeros(p.Mt * p.Nt * p.B_total * p.single_tile_size / sizeof(std::uint32_t), 0);

    constexpr std::uint32_t do_bcast = 0;
    for (std::uint32_t launch = 0; launch < launches; launch++) {
        auto spec = build_bmm_program_spec(p, tensors, node_range, /*use_implicit_sync=*/true);
        tt_metal::Program program(mesh_device, spec);
        ProgramRunArgs params;
        params.kernel_run_args = {
            ProgramRunArgs::KernelRunArgs{
                .kernel = READER,
                .runtime_arg_values = batch_start,
                .common_runtime_arg_values =
                    {{"Mt", p.Mt},
                     {"Kt", p.Kt},
                     {"Nt", p.Nt},
                     {"MtKt", p.Mt * p.Kt},
                     {"KtNt", p.Kt * p.Nt},
                     {"batch", p.B_per_core},
                     {"do_bcast", do_bcast}},
            },
            ProgramRunArgs::KernelRunArgs{
                .kernel = WRITER,
                .runtime_arg_values = batch_start,
                .common_runtime_arg_values = {{"Mt", p.Mt}, {"Nt", p.Nt}, {"batch", p.B_per_core}},
            },
            ProgramRunArgs::KernelRunArgs{.kernel = COMPUTE},
        };
        params.tensor_args = {
            {SRC0_T, TensorArgument{tensors.src0}},
            {SRC1_T, TensorArgument{tensors.src1}},
            {DST_T, TensorArgument{tensors.dst}},
        };
        program.set_program_run_args(params);

        slow_dispatch::WriteToBuffer(tensors.dst.mesh_buffer(), zeros);
        LaunchProgram(mesh_device, std::move(program));

        std::vector<std::uint32_t> result_vec;
        slow_dispatch::ReadFromBuffer(tensors.dst.mesh_buffer(), result_vec);
        int argfail = -1;
        bool pass = validate_bmm_result(p, inputs.src0, inputs.src1, result_vec, &argfail);
        EXPECT_TRUE(pass) << "launch " << launch << ": failure position=" << argfail;
    }
}

TEST_F(QuasarMeshDeviceSingleCardFixture, BmmAllNeosMultinode) {
    BmmParams p;
    p.Mt = 4;
    p.Kt = 4;
    p.Nt = 1;
    p.B_per_core = 1;
    p.num_input_tiles = 16;  // see BmmAllNeos
    p.num_output_tiles = 8;
    run_bmm_all_neos_all_nodes(*devices_[0], p, /*launches=*/1);
}

// BmmAllNeosMultinode scaled up: 2 output rows per NEO, 4 batches per node, 3 launches. Mt and Kt stay
// multiples of 4, the reader count, so each B tile's reader lines up with the NEOs' order. Nt stays 1: each
// writer takes dst entries from its two NEOs in turn, which is the row order only when a row is one tile.
// Kt stays 4: with Kt = 8 (256-term bf16 sums) a few values in a million land just outside the 5% tolerance.
// src0 holds 8 tiles per reader; src1 is sized for one B tile per reader per txn-id batch.
TEST_F(QuasarMeshDeviceSingleCardFixture, BmmAllNeosMultinodeLarge) {
    BmmParams p;
    p.Mt = 8;
    p.Kt = 4;
    p.Nt = 1;
    p.B_per_core = 4;
    p.num_input_tiles = 32;
    p.num_input_tiles_B = 8;
    p.num_output_tiles = 8;
    run_bmm_all_neos_all_nodes(*devices_[0], p, /*launches=*/3);
}

// BmmAllNeosMultinode and BmmAllNeosMultinodeLarge with C written to DRAM as posted writes: the writers' DFB
// credits come from WR_SENT (the write has left the node) rather than from DRAM's acknowledgement.
TEST_F(QuasarMeshDeviceSingleCardFixture, BmmAllNeosMultinodePosted) {
    BmmParams p;
    p.Mt = 4;
    p.Kt = 4;
    p.Nt = 1;
    p.B_per_core = 1;
    p.num_input_tiles = 16;
    p.num_output_tiles = 8;
    p.posted_writes = true;
    run_bmm_all_neos_all_nodes(*devices_[0], p, /*launches=*/1);
}

TEST_F(QuasarMeshDeviceSingleCardFixture, BmmAllNeosMultinodeLargePosted) {
    BmmParams p;
    p.Mt = 8;
    p.Kt = 4;
    p.Nt = 1;
    p.B_per_core = 4;
    p.num_input_tiles = 32;
    p.num_input_tiles_B = 8;
    p.num_output_tiles = 8;
    p.posted_writes = true;
    run_bmm_all_neos_all_nodes(*devices_[0], p, /*launches=*/3);
}

TEST_F(QuasarMeshDeviceSingleCardFixture, TraceBmm) {
    if (this->IsSlowDispatch()) {
        GTEST_SKIP();
    }
    auto& mesh_device = *devices_[0];
    const BmmParams p = single_node_bmm_params(mesh_device.arch());
    BmmTensors tensors = create_bmm_tensors(mesh_device, p);
    BmmSingleNode bmm = setup_bmm_single_node(mesh_device, p, tensors);
    auto& cq = mesh_device.mesh_command_queue();

    // Warm up
    distributed::EnqueueMeshWorkload(cq, bmm.workload, /*blocking=*/true);
    check_bmm_output(cq, p, tensors, bmm.inputs);

    const std::vector<std::uint32_t> zeros(p.single_tile_size * p.Mt * p.Nt * p.B_total / sizeof(std::uint32_t), 0);
    cq.enqueue_write_tensor(HostTensor::from_vector(zeros, tensors.dst.tensor_spec()), tensors.dst);

    distributed::MeshTraceId trace_id = mesh_device.begin_mesh_trace(cq);
    distributed::EnqueueMeshWorkload(cq, bmm.workload, /*blocking=*/false);
    mesh_device.end_mesh_trace(cq, trace_id);

    mesh_device.replay_mesh_trace(cq, trace_id, /*blocking=*/true);
    check_bmm_output(cq, p, tensors, bmm.inputs);

    mesh_device.release_mesh_trace(trace_id);
}
