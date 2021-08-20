import unittest
from pipeline.combineplanemasks import combine_plane_masks
from pipeline.uploadresults import get_compressed_index_mask
import numpy as np
import zlib


class TestPlanePostProcessing(unittest.TestCase):
    def test_combine_plane_masks(self):
        # Setup plane masks
        num_planes = 2
        plane_height = 480
        plane_width = 640

        plane_masks = np.zeros(
            (num_planes, plane_height, plane_width),
            dtype=np.uint8
        )

        plane_masks[0, :plane_height // 2] = 255
        plane_masks[1, plane_height // 2:, :10] = 255

        # Combine call we want to test
        index_mask, alpha_mask = combine_plane_masks(plane_masks)

        # Check shapes
        self.assertSequenceEqual(
            index_mask.shape, (plane_height, plane_width))
        self.assertSequenceEqual(
            alpha_mask.shape, (plane_height, plane_width))

        # Check dtypes
        self.assertEqual(index_mask.dtype, np.int16)
        self.assertEqual(alpha_mask.dtype, np.float32)

        # Check index values
        self.assertTrue(
            np.all(index_mask[:plane_height // 2] == 0))
        self.assertTrue(
            np.all(index_mask[plane_height // 2:, :10] == 1))
        self.assertTrue(
            np.all(index_mask[plane_height // 2:, 10:] == -1))

        # Check alpha values
        self.assertTrue(
            np.all(alpha_mask[:plane_height // 2] == 1))
        self.assertTrue(
            np.all(alpha_mask[plane_height // 2:, :10] == 1))
        self.assertTrue(
            np.all(alpha_mask[plane_height // 2:, 10:] == 0))

    def test_compressed_index_mask(self):
        # Setup plane masks and combine
        num_planes = 2
        plane_height = 480
        plane_width = 640

        plane_masks = np.zeros(
            (num_planes, plane_height, plane_width),
            dtype=np.uint8
        )

        plane_masks[0, :plane_height // 2] = 255
        plane_masks[1, plane_height // 2:, :10] = 255

        index_mask, _ = combine_plane_masks(plane_masks)

        # Compress index mask
        compressed_index_mask = get_compressed_index_mask(index_mask)

        self.assertIsInstance(compressed_index_mask, bytes)

        # Decompress and compare to original
        decompress = zlib.decompressobj()
        decompressed_index_mask = decompress.decompress(compressed_index_mask)

        int16_index_mask_flat = np.frombuffer(
            decompressed_index_mask, dtype=np.int16)
        int16_index_mask = int16_index_mask_flat.reshape(
            (plane_height, plane_width))

        self.assertTrue(np.all(index_mask == int16_index_mask))

        # Test if making the index mask by hand yields same results.
        # This is what we do in JS.
        hand_built_index_mask = []
        for x in range(plane_width):
            col = []
            for y in range(plane_height):
                col.append(int16_index_mask_flat[y * plane_width + x])
            hand_built_index_mask.append(col)

        for x in range(plane_width):
            for y in range(plane_height):
                self.assertEqual(hand_built_index_mask[x][y], int16_index_mask[y, x],
                                 "Hand-made and numpy index mask not equal for x=%d y=%d" % (x, y))


if __name__ == "__main__":
    unittest.main()
