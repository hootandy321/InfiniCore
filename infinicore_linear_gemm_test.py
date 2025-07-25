#!/usr/bin/env python3
"""
InfiniCore Linear vs GEMM Comprehensive Performance Test

This script provides the actual performance comparison test infrastructure
for InfiniCore's Linear and GEMM operators when the library is built.

Usage:
    # After building InfiniCore with: python scripts/install.py --cpu=y --nv-gpu=y
    python infinicore_linear_gemm_test.py --device cpu --iterations 100
    python infinicore_linear_gemm_test.py --device nvidia --iterations 100
"""

import sys
import os
import time
import ctypes
from ctypes import c_uint64
import argparse
import statistics
from typing import List, Dict, Tuple, Optional

# Add the test directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'test', 'infiniop'))

try:
    from libinfiniop import (
        LIBINFINIOP,
        TestTensor,
        get_test_devices,
        check_error,
        TestWorkspace,
        InfiniDtype,
        InfiniDtypeNames,
        InfiniDeviceNames,
        infiniopOperatorDescriptor_t,
    )
    INFINICORE_AVAILABLE = True
    print("✅ InfiniCore bindings loaded successfully")
except (ImportError, AssertionError) as e:
    INFINICORE_AVAILABLE = False
    print(f"❌ InfiniCore not available: {e}")
    print("Please build InfiniCore first: python scripts/install.py --cpu=y --nv-gpu=y")
    exit(1)

class InfiniCorePerformanceTest:
    """Performance test class for InfiniCore operators"""
    
    def __init__(self, device_type: str, dtype: InfiniDtype = InfiniDtype.F32, 
                 num_warmup: int = 10, num_iterations: int = 100):
        self.device_type = device_type
        self.dtype = dtype
        self.num_warmup = num_warmup
        self.num_iterations = num_iterations
        self.results = []
        
    def benchmark_linear_operator(self, handle, batch_size: int, seq_len: int, 
                                 in_features: int, out_features: int, 
                                 has_bias: bool = True) -> Dict:
        """Benchmark InfiniCore Linear operator"""
        
        print(f"  🧠 Testing Linear operator...")
        
        # Initialize tensors
        input_shape = (batch_size * seq_len, in_features)
        weight_shape = (out_features, in_features)
        output_shape = (batch_size * seq_len, out_features)
        bias_shape = (out_features,) if has_bias else None
        
        input_tensor = TestTensor(input_shape, None, self.dtype, self.device_type)
        weight_tensor = TestTensor(weight_shape, None, self.dtype, self.device_type)
        output_tensor = TestTensor(output_shape, None, self.dtype, self.device_type, mode="zeros")
        bias_tensor = TestTensor(bias_shape, None, self.dtype, self.device_type) if has_bias else None
        
        # Create descriptor
        descriptor = infiniopOperatorDescriptor_t()
        bias_desc = bias_tensor.descriptor if has_bias else None
        
        check_error(
            LIBINFINIOP.infiniopCreateLinearDescriptor(
                handle,
                ctypes.byref(descriptor),
                output_tensor.descriptor,
                input_tensor.descriptor,
                weight_tensor.descriptor,
                bias_desc,
            )
        )
        
        # Get workspace
        workspace_size = c_uint64(0)
        check_error(
            LIBINFINIOP.infiniopGetLinearWorkspaceSize(
                descriptor, ctypes.byref(workspace_size)
            )
        )
        workspace = TestWorkspace(workspace_size.value, self.device_type)
        
        # Warmup
        for _ in range(self.num_warmup):
            bias_data = bias_tensor.data() if has_bias else None
            check_error(
                LIBINFINIOP.infiniopLinear(
                    descriptor,
                    workspace.data(),
                    workspace_size.value,
                    output_tensor.data(),
                    input_tensor.data(),
                    weight_tensor.data(),
                    bias_data,
                    None,
                )
            )
        
        # Synchronize if needed
        if self.device_type != 'cpu':
            # Add device synchronization here if needed
            pass
        
        # Benchmark
        times = []
        for _ in range(self.num_iterations):
            start_time = time.perf_counter()
            
            bias_data = bias_tensor.data() if has_bias else None
            check_error(
                LIBINFINIOP.infiniopLinear(
                    descriptor,
                    workspace.data(),
                    workspace_size.value,
                    output_tensor.data(),
                    input_tensor.data(),
                    weight_tensor.data(),
                    bias_data,
                    None,
                )
            )
            
            # Synchronize if needed
            if self.device_type != 'cpu':
                pass  # Add device sync
            
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)
        
        # Cleanup
        check_error(LIBINFINIOP.infiniopDestroyLinearDescriptor(descriptor))
        
        # Calculate metrics
        avg_time = statistics.mean(times)
        min_time = min(times)
        max_time = max(times)
        std_time = statistics.stdev(times) if len(times) > 1 else 0
        
        flops = 2.0 * batch_size * seq_len * in_features * out_features / 1e9
        throughput = flops / (avg_time / 1000)
        
        print(f"     Time: {avg_time:.2f}ms ± {std_time:.2f}ms")
        print(f"     Throughput: {throughput:.1f} GFLOPS")
        
        return {
            'operation': 'InfiniCore Linear',
            'avg_time_ms': avg_time,
            'min_time_ms': min_time,
            'max_time_ms': max_time,
            'std_time_ms': std_time,
            'throughput_gflops': throughput,
            'flops': flops
        }
    
    def benchmark_gemm_operator(self, handle, batch_size: int, seq_len: int,
                               in_features: int, out_features: int,
                               has_bias: bool = True) -> Dict:
        """Benchmark InfiniCore GEMM operator"""
        
        print(f"  🔧 Testing GEMM operator...")
        
        # For GEMM equivalent to Linear: input @ weight.T + bias
        # We need: C = 1.0 * A @ B + 0.0 * C (initially zeros)
        # Then: C = C + bias (separate operation or set beta=1, C=bias initially)
        
        # Initialize tensors
        a_shape = (batch_size * seq_len, in_features)  # input
        b_shape = (in_features, out_features)  # weight.T
        c_shape = (batch_size * seq_len, out_features)  # output
        
        a_tensor = TestTensor(a_shape, None, self.dtype, self.device_type)
        b_tensor = TestTensor(b_shape, None, self.dtype, self.device_type)
        c_tensor = TestTensor(c_shape, None, self.dtype, self.device_type, mode="zeros")
        
        # Create descriptor
        descriptor = infiniopOperatorDescriptor_t()
        check_error(
            LIBINFINIOP.infiniopCreateGemmDescriptor(
                handle,
                ctypes.byref(descriptor),
                c_tensor.descriptor,
                a_tensor.descriptor,
                b_tensor.descriptor,
            )
        )
        
        # Get workspace
        workspace_size = c_uint64(0)
        check_error(
            LIBINFINIOP.infiniopGetGemmWorkspaceSize(
                descriptor, ctypes.byref(workspace_size)
            )
        )
        workspace = TestWorkspace(workspace_size.value, self.device_type)
        
        # Prepare bias tensor for separate addition if needed
        bias_tensor = TestTensor((out_features,), None, self.dtype, self.device_type) if has_bias else None
        
        # Warmup
        for _ in range(self.num_warmup):
            # GEMM: C = 1.0 * A @ B + 0.0 * C
            check_error(
                LIBINFINIOP.infiniopGemm(
                    descriptor,
                    workspace.data(),
                    workspace_size.value,
                    c_tensor.data(),
                    a_tensor.data(),
                    b_tensor.data(),
                    1.0,  # alpha
                    0.0,  # beta
                    None,
                )
            )
            
            # Add bias (simulated - in practice this would be another operation)
            if has_bias:
                # This would require additional bias addition operation
                pass
        
        # Synchronize if needed
        if self.device_type != 'cpu':
            pass
        
        # Benchmark
        times = []
        for _ in range(self.num_iterations):
            start_time = time.perf_counter()
            
            # GEMM operation
            check_error(
                LIBINFINIOP.infiniopGemm(
                    descriptor,
                    workspace.data(),
                    workspace_size.value,
                    c_tensor.data(),
                    a_tensor.data(),
                    b_tensor.data(),
                    1.0,  # alpha
                    0.0,  # beta  
                    None,
                )
            )
            
            # Add bias (would be separate operation in practice)
            if has_bias:
                # Bias addition time would be included here
                pass
            
            # Synchronize if needed
            if self.device_type != 'cpu':
                pass
            
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)
        
        # Cleanup
        check_error(LIBINFINIOP.infiniopDestroyGemmDescriptor(descriptor))
        
        # Calculate metrics
        avg_time = statistics.mean(times)
        min_time = min(times)
        max_time = max(times)
        std_time = statistics.stdev(times) if len(times) > 1 else 0
        
        flops = 2.0 * batch_size * seq_len * in_features * out_features / 1e9
        throughput = flops / (avg_time / 1000)
        
        print(f"     Time: {avg_time:.2f}ms ± {std_time:.2f}ms")
        print(f"     Throughput: {throughput:.1f} GFLOPS")
        
        return {
            'operation': 'InfiniCore GEMM',
            'avg_time_ms': avg_time,
            'min_time_ms': min_time,
            'max_time_ms': max_time,
            'std_time_ms': std_time,
            'throughput_gflops': throughput,
            'flops': flops
        }
    
    def run_comprehensive_test(self, handle) -> List[Dict]:
        """Run comprehensive test across different configurations"""
        
        test_configs = [
            # (batch_size, seq_len, in_features, out_features, description)
            (1, 128, 256, 1024, "Small transformer layer"),
            (4, 128, 256, 1024, "Small transformer (batch=4)"),
            (1, 512, 768, 3072, "BERT-base FFN (single)"),
            (8, 512, 768, 3072, "BERT-base FFN (batch=8)"),
            (1, 2048, 4096, 11008, "LLaMA-7B FFN (single)"),
            (4, 2048, 4096, 11008, "LLaMA-7B FFN (batch=4)"),
        ]
        
        print(f"\n{'='*80}")
        print(f"INFINICORE PERFORMANCE TEST - {InfiniDeviceNames.get(self.device_type, self.device_type).upper()}")
        print(f"{'='*80}")
        print(f"Data type: {InfiniDtypeNames.get(self.dtype, str(self.dtype))}")
        print(f"Warmup iterations: {self.num_warmup}")
        print(f"Benchmark iterations: {self.num_iterations}")
        print()
        
        all_results = []
        
        for batch_size, seq_len, in_features, out_features, description in test_configs:
            print(f"🔍 {description}")
            print(f"   Shape: batch={batch_size}, seq_len={seq_len}, in_features={in_features}, out_features={out_features}")
            
            # Skip very large configurations on CPU
            if self.device_type == 'cpu' and in_features > 2048:
                print("   ⏭️  Skipping large configuration on CPU")
                continue
            
            try:
                # Test Linear operator
                linear_result = self.benchmark_linear_operator(
                    handle, batch_size, seq_len, in_features, out_features
                )
                linear_result['config'] = description
                linear_result['batch_size'] = batch_size
                linear_result['seq_len'] = seq_len
                linear_result['in_features'] = in_features
                linear_result['out_features'] = out_features
                all_results.append(linear_result)
                
                # Test GEMM operator
                gemm_result = self.benchmark_gemm_operator(
                    handle, batch_size, seq_len, in_features, out_features
                )
                gemm_result['config'] = description
                gemm_result['batch_size'] = batch_size
                gemm_result['seq_len'] = seq_len
                gemm_result['in_features'] = in_features
                gemm_result['out_features'] = out_features
                all_results.append(gemm_result)
                
                # Calculate speedup
                speedup = gemm_result['avg_time_ms'] / linear_result['avg_time_ms']
                print(f"   🚀 Linear vs GEMM speedup: {speedup:.2f}x")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
            
            print()
        
        self.results.extend(all_results)
        return all_results
    
    def generate_report(self) -> str:
        """Generate performance report"""
        
        if not self.results:
            return "No benchmark results available."
        
        report = []
        report.append("=" * 80)
        report.append("INFINICORE LINEAR VS GEMM PERFORMANCE REPORT")
        report.append("=" * 80)
        report.append()
        
        # Group results by configuration
        configs = {}
        for result in self.results:
            config = result['config']
            if config not in configs:
                configs[config] = {}
            configs[config][result['operation']] = result
        
        # Performance comparison table
        report.append(f"{'Configuration':<25} {'Linear (ms)':<12} {'GEMM (ms)':<12} {'Speedup':<10} {'Linear GFLOPS':<15}")
        report.append("-" * 80)
        
        speedups = []
        for config, ops in configs.items():
            if 'InfiniCore Linear' in ops and 'InfiniCore GEMM' in ops:
                linear = ops['InfiniCore Linear']
                gemm = ops['InfiniCore GEMM']
                speedup = gemm['avg_time_ms'] / linear['avg_time_ms']
                speedups.append(speedup)
                
                report.append(f"{config:<25} {linear['avg_time_ms']:<12.2f} "
                            f"{gemm['avg_time_ms']:<12.2f} {speedup:<10.2f}x "
                            f"{linear['throughput_gflops']:<15.1f}")
        
        report.append("")
        
        # Summary statistics
        if speedups:
            avg_speedup = statistics.mean(speedups)
            max_speedup = max(speedups)
            min_speedup = min(speedups)
            
            report.append("PERFORMANCE SUMMARY:")
            report.append(f"  Average Linear speedup: {avg_speedup:.2f}x")
            report.append(f"  Best case speedup: {max_speedup:.2f}x")
            report.append(f"  Worst case speedup: {min_speedup:.2f}x")
            report.append(f"  Performance range: {(max_speedup - min_speedup):.2f}x")
            report.append("")
            
            # Conclusion
            if avg_speedup > 1.15:
                report.append("✅ CONCLUSION: Linear operator shows significant performance advantage")
                report.append("   Recommendation: Use Linear operator for neural network layers")
            elif avg_speedup > 1.05:
                report.append("✅ CONCLUSION: Linear operator shows measurable performance advantage")
                report.append("   Recommendation: Use Linear operator for performance and convenience")
            else:
                report.append("⚖️  CONCLUSION: Similar performance between Linear and GEMM")
                report.append("   Recommendation: Use Linear operator for API convenience")
        
        return "\n".join(report)

def main():
    """Main function"""
    
    if not INFINICORE_AVAILABLE:
        print("❌ InfiniCore not available. Please build the project first.")
        return
    
    parser = argparse.ArgumentParser(description="InfiniCore Linear vs GEMM Performance Test")
    parser.add_argument("--device", choices=['cpu', 'nvidia', 'ascend', 'cambricon'], 
                       default='cpu', help="Device to test on")
    parser.add_argument("--dtype", choices=['f16', 'bf16', 'f32'], 
                       default='f32', help="Data type to test")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations")
    parser.add_argument("--iterations", type=int, default=100, help="Number of benchmark iterations")
    parser.add_argument("--output", type=str, help="Output file for report")
    
    args = parser.parse_args()
    
    # Map device and dtype
    device_map = {
        'cpu': 'cpu',
        'nvidia': 'nvidia', 
        'ascend': 'ascend',
        'cambricon': 'cambricon'
    }
    
    dtype_map = {
        'f16': InfiniDtype.F16,
        'bf16': InfiniDtype.BF16,
        'f32': InfiniDtype.F32
    }
    
    device_type = device_map[args.device]
    dtype = dtype_map[args.dtype]
    
    # Initialize test
    test = InfiniCorePerformanceTest(
        device_type=device_type,
        dtype=dtype,
        num_warmup=args.warmup,
        num_iterations=args.iterations
    )
    
    # Get device handle (this would need to be implemented based on actual InfiniCore API)
    # handle = get_device_handle(device_type)
    handle = None  # Placeholder - implement based on actual API
    
    try:
        # Run comprehensive test
        results = test.run_comprehensive_test(handle)
        
        # Generate and display report
        report = test.generate_report()
        print(report)
        
        # Save report if requested
        if args.output:
            with open(args.output, 'w') as f:
                f.write(report)
            print(f"\n📄 Report saved to: {args.output}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("Please ensure InfiniCore is properly built and configured.")

if __name__ == "__main__":
    main()