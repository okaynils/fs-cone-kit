import unittest

from core.benchmarking import summarize_latencies


class BenchmarkSummaryTests(unittest.TestCase):
    def test_reports_median_p95_and_throughput(self):
        result = summarize_latencies([0.01, 0.02, 0.03, 0.04], batch_size=2)
        self.assertAlmostEqual(result["median_batch_latency_ms"], 25.0)
        self.assertAlmostEqual(result["p95_batch_latency_ms"], 40.0)
        self.assertAlmostEqual(result["median_latency_ms_per_image"], 12.5)
        self.assertAlmostEqual(result["throughput_images_per_second"], 80.0)


if __name__ == "__main__":
    unittest.main()
