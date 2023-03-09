import torch
import unittest
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current)
sys.path.append(parent_directory)

from utils import RunningTensor
from Memory import Memory

class TestMemory(unittest.TestCase):
    def setUp(self):
        self.memory = Memory()
        self.embeddings1 = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)
        self.embeddings2 = torch.tensor([[10, 11, 12], [13, 14, 15], [16, 17, 18]], dtype=torch.float32)
        self.label1 = 0
        self.label2 = 1

    def test_push_new_class(self):
        self.memory.push(self.label1, self.embeddings1)
        self.assertIn(self.label1, self.memory.centroids)
        self.assertEqual(self.memory.centroids[self.label1].count, 1)
        self.assertTrue(torch.equal(self.memory.centroids[self.label1].mean, self.embeddings1.mean(dim=0)))

    def test_push_existing_class(self):
        self.memory.push(self.label1, self.embeddings1)
        self.memory.push(self.label1, self.embeddings2)
        self.assertIn(self.label1, self.memory.centroids)
        self.assertEqual(self.memory.centroids[self.label1].count, 2)
        self.assertTrue(torch.equal(self.memory.centroids[self.label1].mean, torch.tensor([5.5, 6.5, 7.5])))

    def test_push_multiple_classes(self):
        self.memory.push(self.label1, self.embeddings1)
        self.memory.push(self.label2, self.embeddings2)
        self.assertIn(self.label1, self.memory.centroids)
        self.assertEqual(self.memory.centroids[self.label1].count, 1)
        self.assertTrue(torch.equal(self.memory.centroids[self.label1].mean, self.embeddings1.mean(dim=0)))
        self.assertIn(self.label2, self.memory.centroids)
        self.assertEqual(self.memory.centroids[self.label2].count, 1)
        self.assertTrue(torch.equal(self.memory.centroids[self.label2].mean, self.embeddings2.mean(dim=0)))

if __name__ == '__main__':
    unittest.main()