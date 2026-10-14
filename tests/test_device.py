from wispo.asr.device import parse_sycl_ls


def test_parse_sycl_ls_reads_the_arc_gpu_name():
    text = (
        "[level_zero:gpu][level_zero:0] Intel(R) oneAPI Unified Runtime over Level-Zero, "
        "Intel(R) Arc(TM) Graphics 12.74.4 [1.14.37020]\n"
        "[opencl:cpu][opencl:0] Intel(R) OpenCL, Intel(R) Core(TM) Ultra 5 225H\n"
    )
    assert parse_sycl_ls(text) == "Intel(R) Arc(TM) Graphics"


def test_parse_sycl_ls_ignores_cpu_only_output():
    text = "[opencl:cpu][opencl:0] Intel(R) OpenCL, Intel(R) Core(TM) Ultra 5 225H\n"
    assert parse_sycl_ls(text) is None
