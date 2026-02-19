"""Unit tests for novel DCMDA enhancements."""
import sys
import os
import unittest
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')


class TestEdgeDropout(unittest.TestCase):
    """Tests for the edge_dropout graph augmentation function."""

    def test_edge_dropout_preserves_shape(self):
        """Edge dropout should not change matrix dimensions."""
        from main import edge_dropout
        adj = np.eye(10)
        result = edge_dropout(adj, drop_rate=0.1)
        self.assertEqual(result.shape, adj.shape)

    def test_edge_dropout_zero_rate(self):
        """With drop_rate=0, no edges should be removed."""
        from main import edge_dropout
        adj = np.ones((5, 5))
        result = edge_dropout(adj, drop_rate=0.0)
        np.testing.assert_array_equal(result, adj)

    def test_edge_dropout_reduces_edges(self):
        """Edge dropout should reduce or maintain edge count."""
        from main import edge_dropout
        np.random.seed(42)
        adj = np.ones((10, 10))
        result = edge_dropout(adj, drop_rate=0.5)
        self.assertLessEqual(np.sum(result > 0), np.sum(adj > 0))

    def test_edge_dropout_does_not_add_edges(self):
        """Edge dropout should never add new edges."""
        from main import edge_dropout
        np.random.seed(42)
        adj = np.eye(10)
        result = edge_dropout(adj, drop_rate=0.1)
        zero_mask = adj == 0
        np.testing.assert_array_equal(result[zero_mask], 0)

    def test_edge_dropout_original_unchanged(self):
        """Edge dropout should not modify the original matrix."""
        from main import edge_dropout
        adj = np.ones((5, 5))
        original = adj.copy()
        edge_dropout(adj, drop_rate=0.5)
        np.testing.assert_array_equal(adj, original)


class TestMultiHeadCrossAttention(unittest.TestCase):
    """Tests for the MultiHeadCrossAttention layer."""

    def test_output_shape(self):
        """Output shape should match input shape."""
        from classifiers import MultiHeadCrossAttention
        layer = MultiHeadCrossAttention(num_heads=4, key_dim=16)
        query = tf.random.normal((2, 1, 64))
        key_value = tf.random.normal((2, 1, 64))
        output = layer([query, key_value])
        self.assertEqual(output.shape, (2, 1, 64))

    def test_different_num_heads(self):
        """Layer should work with different numbers of heads."""
        from classifiers import MultiHeadCrossAttention
        for num_heads in [1, 2, 8]:
            layer = MultiHeadCrossAttention(num_heads=num_heads, key_dim=8)
            query = tf.random.normal((1, 1, 32))
            kv = tf.random.normal((1, 1, 32))
            output = layer([query, kv])
            self.assertEqual(output.shape, (1, 1, 32))

    def test_get_config(self):
        """get_config should return layer parameters."""
        from classifiers import MultiHeadCrossAttention
        layer = MultiHeadCrossAttention(num_heads=4, key_dim=16)
        config = layer.get_config()
        self.assertEqual(config['num_heads'], 4)
        self.assertEqual(config['key_dim'], 16)


class TestGCNFix(unittest.TestCase):
    """Tests for the fixed GraphConvolution layer."""

    def test_class_has_build_method(self):
        """GraphConvolution should have build as an instance method."""
        from GCN import GraphConvolution
        layer = GraphConvolution(units=32)
        self.assertTrue(hasattr(layer, 'build'))
        self.assertTrue(callable(layer.build))

    def test_class_has_call_method(self):
        """GraphConvolution should have call as an instance method."""
        from GCN import GraphConvolution
        layer = GraphConvolution(units=32)
        self.assertTrue(hasattr(layer, 'call'))
        self.assertTrue(callable(layer.call))

    def test_class_has_activation(self):
        """GraphConvolution should store the activation function."""
        from GCN import GraphConvolution
        layer = GraphConvolution(units=32, activation='relu')
        self.assertIsNotNone(layer.activation)


class TestNMFConvergence(unittest.TestCase):
    """Tests for improved NMF with convergence checking."""

    def test_nmf_output_shapes(self):
        """NMF should return matrices with correct dimensions."""
        from NMF import get_low_feature
        np.random.seed(42)
        A = np.random.rand(10, 5)
        A[A < 0.7] = 0
        k = 3
        U, V = get_low_feature(k, 0.001, 1e-4, A)
        self.assertEqual(U.shape, (10, 3))
        self.assertEqual(V.shape, (5, 3))

    def test_nmf_non_negative(self):
        """NMF outputs should be non-negative."""
        from NMF import get_low_feature
        np.random.seed(42)
        A = np.random.rand(10, 5)
        U, V = get_low_feature(3, 0.001, 1e-4, A)
        self.assertTrue(np.all(U >= 0))
        self.assertTrue(np.all(V >= 0))


class TestMetric(unittest.TestCase):
    """Tests for metric computation."""

    def test_get_metrics_perfect(self):
        """Perfect predictions should give high metrics."""
        from metric import get_metrics
        real = np.array([1, 1, 1, 0, 0, 0])
        pred = np.array([0.9, 0.8, 0.7, 0.1, 0.2, 0.15])
        metrics = get_metrics(real, pred)
        self.assertEqual(len(metrics), 7)
        self.assertGreater(metrics[0], 0.9)  # AUPR
        self.assertGreater(metrics[1], 0.9)  # AUC

    def test_get_metrics_returns_list(self):
        """get_metrics should return a list of 7 values."""
        from metric import get_metrics
        real = np.array([1, 0, 1, 0])
        pred = np.array([0.6, 0.4, 0.7, 0.3])
        metrics = get_metrics(real, pred)
        self.assertEqual(len(metrics), 7)


if __name__ == '__main__':
    unittest.main()
