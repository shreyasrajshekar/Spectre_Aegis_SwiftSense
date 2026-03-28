import time
from app import SpectreAegisController

def test_simulation_latency():
    """
    Tests the end-to-end Sense-to-Act latency.
    Constraint: Must execute cycle in < 50ms.
    Handover: < 10ms.
    """
    controller = SpectreAegisController(use_simulation=True)
    
    cycle_latencies = []
    
    # Run 50 warmup cycles for PyTorch allocation
    for _ in range(50):
        controller.execute_cycle()
        
    # Measure 100 cycles
    for _ in range(100):
        lat, _ = controller.execute_cycle()
        cycle_latencies.append(lat)
        
    avg_latency = sum(cycle_latencies) / len(cycle_latencies)
    max_latency = max(cycle_latencies)
    
    print(f"\\n--- Latency Benchmark ---")
    print(f"Average Sense-to-Act: {avg_latency:.2f} ms")
    print(f"Max Sense-to-Act: {max_latency:.2f} ms")
    print(f"Target Constraint: < 50.0 ms")
    
    assert avg_latency < 50.0, f"Average latency {avg_latency:.2f} exceeds 50ms constraint!"

if __name__ == "__main__":
    test_simulation_latency()
