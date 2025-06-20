using System.Collections.Generic;
using System.Diagnostics.Metrics;
using System.Threading.Tasks;
using TorchSharp;
using static SamSharp.Utils.Classes;
using static Tensorboard.CostGraphDef.Types;
using static Tensorboard.TensorShapeProto.Types;
using static TorchSharp.torch;

namespace SamSharp.Utils
{
	internal class Amg
	{
		/// <summary>
		/// Generates point grids for all crop layers.
		/// </summary>
		/// <param name="n_per_side"></param>
		/// <param name="n_layers"></param>
		/// <param name="scale_per_layer"></param>
		/// <param name=""></param>
		/// <returns></returns>
		internal static List<Tensor> build_all_layer_point_grids(int n_per_side, int n_layers, int scale_per_layer)
		{
			List<Tensor> points_by_layer = new List<Tensor>();
			for (int i = 0; i < n_layers + 1; i++)
			{
				int n_points = (int)(n_per_side / (Math.Pow(scale_per_layer, i)));
				points_by_layer.Add(build_point_grid(n_points));
			}
			return points_by_layer;
		}

		/// <summary>
		/// Generates a 2D grid of points evenly spaced in [0,1]x[0,1].
		/// </summary>
		/// <param name="n_per_side"></param>
		/// <returns></returns>
		internal static Tensor build_point_grid(int n_per_side)
		{
			using var _ = NewDisposeScope();
			float offset = 1 / (2 * n_per_side);
			Tensor points_one_side = torch.linspace(offset, 1 - offset, n_per_side);
			Tensor points_x = torch.tile(points_one_side[TensorIndex.None, ..], new long[] { n_per_side, 1 });
			Tensor points_y = torch.tile(points_one_side[.., TensorIndex.None], new long[] { 1, n_per_side });
			Tensor points = torch.stack(new Tensor[] { points_x, points_y }, dim: -1).reshape(-1, 2);
			return points.MoveToOuterDisposeScope();
		}

		/// <summary>
		///  Computes the stability score for a batch of masks. The stability		score is the IoU between the binary masks obtained by thresholding		the predicted mask logits at high and low values.
		/// </summary>
		internal static Tensor calculate_stability_score(Tensor masks, float mask_threshold, float threshold_offset)
		{
			using var _ = NewDisposeScope();
			// One mask is always contained inside the other.
			// Save memory by preventing unnecessary cast to torch.int64
			Tensor intersections = (
					(masks > (mask_threshold + threshold_offset))
					.sum(-1, type: torch.int16)
					.sum(-1, type: torch.int32)
				);

			Tensor unions = (
					(masks > (mask_threshold - threshold_offset))
					.sum(-1, type: torch.int16)
					.sum(-1, type: torch.int32)
				);

			return (intersections / unions).MoveToOuterDisposeScope();
		}

		/// <summary>
		/// Filter masks at the edge of a crop, but not at the edge of the original image.
		/// </summary>
		/// <param name="boxes"></param>
		/// <param name="crop_box"></param>
		/// <param name="ororig_box"></param>
		/// <param name="atol"></param>
		/// <returns></returns>
		internal static Tensor is_box_near_crop_edge(Tensor boxes, int[] crop_box, int[] orig_box, float atol = 20.0f)
		{
			using var _ = NewDisposeScope();
			Tensor crop_box_torch = torch.as_tensor(crop_box, dtype: torch.float32, device: boxes.device);
			Tensor orig_box_torch = torch.as_tensor(orig_box, dtype: torch.float32, device: boxes.device);
			boxes = uncrop_boxes_xyxy(boxes, crop_box).@float();
			Tensor near_crop_edge = torch.isclose(boxes, crop_box_torch[TensorIndex.None, ..], atol: atol, rtol: 0);
			Tensor near_image_edge = torch.isclose(boxes, orig_box_torch[TensorIndex.None, ..], atol: atol, rtol: 0);
			near_crop_edge = torch.logical_and(near_crop_edge, ~near_image_edge);
			return torch.any(near_crop_edge, dim: 1).MoveToOuterDisposeScope();
		}

		internal static Tensor uncrop_boxes_xyxy(Tensor boxes, int[] crop_box)
		{
			int x0 = crop_box[0];
			int y0 = crop_box[1];
			Tensor offset = torch.tensor(new int[,] { { x0, y0, x0, y0 } }, device: boxes.device);
			// Check if boxes has a channel dimension
			if (boxes.shape.Length == 3)
			{
				offset = offset.unsqueeze(1);
			}

			return boxes + offset;
		}

		internal static Tensor uncrop_masks(Tensor masks, int[] crop_box, int orig_h, int orig_w)
		{
			int x0 = crop_box[0];
			int y0 = crop_box[1];
			int x1 = crop_box[2];
			int y1 = crop_box[3];

			if (x0 == 0 && y0 == 0 && x1 == orig_w && y1 == orig_h)
			{
				return masks;
			}

			// Coordinate transform masks
			int pad_x = orig_w - (x1 - x0);
			int pad_y = orig_h - (y1 - y0);
			long[] pad = new long[] { x0, pad_x - x0, y0, pad_y - y0 };
			return torch.nn.functional.pad(masks, pad, value: 0);
		}

		/// <summary>
		/// Encodes masks to an uncompressed RLE, in the format expected by	pycoco tools.
		/// </summary>
		/// <param name="tensor"></param>
		/// <returns></returns>
		internal static List<Rle> mask_to_rle_pytorch(Tensor tensor)
		{
			using var _ = NewDisposeScope();
			// Put in fortran order and flatten h,w
			long b = tensor.shape[0];
			long h = tensor.shape[1];
			long w = tensor.shape[2];
			tensor = tensor.permute(0, 2, 1).flatten(1);

			// Compute change indices
			Tensor diff = tensor[.., 1..] ^ tensor[.., ..(int)(tensor.shape[1] - 1)];
			Tensor change_indices = diff.nonzero();

			// Encode run length
			List<Rle> @out = new List<Rle>();
			for (int i = 0; i < b; i++)
			{
				Tensor cur_idxs = change_indices[change_indices[.., 0] == i, 1];

				cur_idxs = torch.cat(new Tensor[] {

					torch.tensor(new long []{ 0 }, dtype : cur_idxs.dtype, device : cur_idxs.device),
					cur_idxs + 1,
					torch.tensor(new long []{ h * w }, dtype : cur_idxs.dtype, device : cur_idxs.device),
				});

				Tensor btw_idxs = cur_idxs[1..] - cur_idxs[..(int)(cur_idxs.shape[0] - 1)];

				List<long> counts = (tensor[i, 0].ToSingle() == 0) ? new List<long> { 0 } : new List<long>();
				counts.AddRange(btw_idxs.data<long>().ToArray());


				@out.Add(new Rle { Size = new int[] { (int)h, (int)w }, Counts = counts });
			}
			return @out;
		}

		internal static Tensor batched_nms(Tensor boxes, Tensor scores, Tensor idxs, float iou_threshold)
		{
			return _batched_nms_coordinate_trick(boxes, scores, idxs, iou_threshold);
		}

		private static Tensor _batched_nms_coordinate_trick(Tensor boxes, Tensor scores, Tensor idxs, float iou_threshold)
		{
			if (boxes.numel() == 0)
			{
				return torch.empty(0, dtype: torch.int64, device: boxes.device);
			}

			Tensor max_coordinate = boxes.max();
			Tensor offsets = idxs.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes));
			Tensor boxes_for_nms = boxes + offsets[.., TensorIndex.None];
			Tensor keep = torchvision.ops.nms(boxes_for_nms, scores, iou_threshold);
			return keep;
		}
	}
}
