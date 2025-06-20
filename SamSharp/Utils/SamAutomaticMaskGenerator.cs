using SamSharp.Modeling;
using SkiaSharp;
using TorchSharp;
using static SamSharp.Utils.Classes;
using static TorchSharp.torch;

namespace SamSharp.Utils
{
	public class SamAutomaticMaskGenerator
	{
		private readonly int points_per_side = 32;
		private readonly int points_per_batch = 64;
		private readonly float pred_iou_thresh = 0.88f;
		private readonly float stability_score_thresh = 0.95f;
		private readonly float stability_score_offset = 1.0f;
		private readonly float box_nms_thresh = 0.7f;
		private readonly int crop_n_layers = 0;
		private readonly float crop_nms_thresh = 0.7f;
		private readonly float crop_overlap_ratio = 512.0f / 1500;
		private readonly int crop_n_points_downscale_factor = 1;
		// point_grids: Optional[List[np.ndarray]] = None,
		private readonly List<Tensor> point_grids;
		private readonly int min_mask_region_area = 0;
		private readonly OutputMode output_mode = OutputMode.BinaryMask;

		private readonly Sam model;
		private readonly Device device;
		private readonly ScalarType dtype;


		/// <summary>
		/// Using a SAM model, generates masks for the entire image. Generates a grid of point prompts over the image, then filters	low quality and duplicate masks.The default settings are chosen for SAM with a ViT-H backbone.
		/// </summary>
		/// <param name="checkpointPath">The checkpoint path</param>
		/// <param name="device">Cuda or CPU</param>
		/// <param name="points_per_side">The number of points to be sampled along one side of the image.The total number of points is points_per_side**2. If None, 'point_grids' must provide explicit	point sampling.</param>
		/// <param name="points_per_batch">Sets the number of points run simultaneously	by the model.Higher numbers may be faster but use more GPU memory.</param>
		/// <param name="pred_iou_thresh">A filtering threshold in [0,1], using the	model's predicted mask quality.</param>
		/// <param name="stability_score_thresh">A filtering threshold in [0,1], using the stability of the mask under changes to the cutoff used to binarize the model's mask predictions.</param>
		/// <param name="stability_score_offset">The amount to shift the cutoff when calculated the stability score.</param>
		/// <param name="box_nms_thresh">The box IoU cutoff used by non-maximal suppression to filter duplicate masks.</param>
		/// <param name="crop_n_layers">If > 0, mask prediction will be run again on	crops of the image.Sets the number of layers to run, where each	layer has 2**i_layer number of image crops.</param>
		/// <param name="crop_nms_thresh">The box IoU cutoff used by non-maximal suppression to filter duplicate masks between different crops.</param>
		/// <param name="crop_overlap_ratio">Sets the degree to which crops overlap. In the first crop layer, crops will overlap by this fraction of the image length.Later layers with more crops scale down this overlap.</param>
		/// <param name="crop_n_points_downscale_factor">The number of points-per-side sampled in layer n is scaled down by crop_n_points_downscale_factor**n.</param>
		/// <param name="point_grids">A list over explicit grids of points used for sampling, normalized to[0, 1]. The nth grid in the list is used in the nth crop layer.Exclusive with points_per_side.</param>
		/// <param name="min_mask_region_area">If >0, postprocessing will be applied to remove disconnected regions and holes in masks with area smaller than min_mask_region_area.</param>
		/// <param name="output_mode">The form masks are returned in. Can be 'binary_mask', 'uncompressed_rle', or 'coco_rle'. 'coco_rle' requires pycocotools.	For large resolutions, 'binary_mask' may consume large amounts of memory.</param>
		public SamAutomaticMaskGenerator(string checkpointPath, int points_per_side = 32, int points_per_batch = 64, float pred_iou_thresh = 0.88f, float stability_score_thresh = 0.95f, float stability_score_offset = 1.0f, float box_nms_thresh = 0.7f, int crop_n_layers = 0, float crop_nms_thresh = 0.7f, float crop_overlap_ratio = 512.0f / 1500, int crop_n_points_downscale_factor = 1, List<Tensor>? point_grids = null, int min_mask_region_area = 0, OutputMode output_mode = OutputMode.BinaryMask, SamDevice device = SamDevice.CPU, SamScalarType dtype = SamScalarType.Float32)
		{
			torchvision.io.DefaultImager = new torchvision.io.SkiaImager(100);
			this.device = new Device((DeviceType)device);
			this.dtype = (ScalarType)dtype;
			this.model = BuildSam.BuildSamModel(checkpointPath, this.device, this.dtype);

			bool use_points_per_side = (points_per_side > 0);
			bool use_point_grids = (point_grids is not null);

			if (!(!(use_points_per_side && use_point_grids) && (use_points_per_side || use_point_grids)))
			{
				throw new ArgumentException("Exactly one of points_per_side or point_grid must be provided.");
			}
			if (points_per_side > 0)
			{
				this.point_grids = Amg.build_all_layer_point_grids(points_per_side, crop_n_layers, crop_n_points_downscale_factor);
			}
			else if (point_grids is not null)
			{
				this.point_grids = point_grids;
			}
			else
			{
				throw new ArgumentException("Can't have both points_per_side and point_grid be None.");
			}

			this.model = BuildSam.BuildSamModel(checkpointPath, this.device, this.dtype);
			this.points_per_batch = points_per_batch;
			this.pred_iou_thresh = pred_iou_thresh;
			this.stability_score_thresh = stability_score_thresh;
			this.stability_score_offset = stability_score_offset;
			this.box_nms_thresh = box_nms_thresh;
			this.crop_n_layers = crop_n_layers;
			this.crop_nms_thresh = crop_nms_thresh;
			this.crop_overlap_ratio = crop_overlap_ratio;
			this.crop_n_points_downscale_factor = crop_n_points_downscale_factor;
			this.min_mask_region_area = min_mask_region_area;
			this.output_mode = output_mode;
		}


		public List<PredictOutput> generate(SKBitmap image, int[] crop_box = null, int maxImageSize = 512)
		{
			using var _ = no_grad();
			using var __ = NewDisposeScope();
			model.eval();
			Tensor imgTensor = Tools.ImageTools.GetTensorFromImage(image); long orig_w = imgTensor.shape[2];
			long orig_h = imgTensor.shape[1];
			float scaleFactor = Math.Min((float)maxImageSize / orig_w, (float)maxImageSize / orig_h);
			int newW = (int)Math.Ceiling(orig_w * scaleFactor / 4) * 4;
			int newH = (int)Math.Ceiling(orig_h * scaleFactor / 4) * 4;
			imgTensor = torchvision.transforms.functional.resize(imgTensor, newH, newW).unsqueeze(0);
			long[] original_size = new long[] { orig_h, orig_w };

			float xStart = (float)newW / points_per_batch;
			float yStep = (float)newH / points_per_side;
			float yStart = yStep / 2;

			MaskData data = new MaskData();

			List<BatchedOutput> outputs = new List<BatchedOutput>();

			for (int y = 0; y < this.points_per_side / 2; y++)
			{
				Tensor points = torch.zeros(new long[] { points_per_batch, 2 });
				points[..(this.points_per_batch / 2), 0] = torch.linspace(xStart, newW - xStart, points_per_batch / 2);
				points[(this.points_per_batch / 2).., 0] = torch.linspace(xStart, newW - xStart, points_per_batch / 2);
				points[..(this.points_per_batch / 2), 1] = yStart + yStep * (y * 2 + 0);
				points[(this.points_per_batch / 2).., 1] = yStart + yStep * (y * 2 + 1);
				Tensor labels = torch.ones(points.shape[0]);
				BatchedInput batched = new BatchedInput { Image = imgTensor, Point_coords = points, Point_labels = labels, Original_size = original_size };
				MaskData tempData = _process_batch(batched, crop_box, orig_w, orig_h);
				data.Concat(tempData);
				GC.Collect();
			}

			Tensor keep_by_nms = Amg.batched_nms(
				data.Boxes,
				data.IouPreds,
				torch.zeros_like(data.Boxes[.., 0]),  // categories
				iou_threshold: this.box_nms_thresh);

			data.Filter(keep_by_nms);
			List<PredictOutput> predictOutputs = new List<PredictOutput>();
			for (int i = 0; i < data.IouPreds.shape[0]; i++)
			{
				bool[,] maskArray = new bool[data.Masks.shape[2], data.Masks.shape[1]];
				var arrayData = data.Masks.transpose(2, 1)[i].data<bool>().ToArray();
				Buffer.BlockCopy(arrayData, 0, maskArray, 0, arrayData.Length * sizeof(bool));

				predictOutputs.Add(new PredictOutput
				{
					Mask = maskArray,
					Precision = data.IouPreds[i].ToSingle(),
				});

			}
			return predictOutputs;
		}


		internal BatchedOutput _process_eproch(BatchedInput batched, int[] crop_box, long orig_w, long orig_h)
		{
			using var _ = no_grad();
			using var __ = NewDisposeScope();

			Tensor points = batched.Point_coords;
			batched.Point_coords = batched.Point_coords[.., TensorIndex.None, ..];
			batched.Point_labels = batched.Point_labels[.., TensorIndex.None];

			BatchedOutput output = model.forward(batched, true, true);
			output.to(CPU);
			output.Masks = output.Masks.MoveToOuterDisposeScope();
			output.Iou_predictions = output.Iou_predictions.MoveToOuterDisposeScope();
			output.Low_res_logits = output.Low_res_logits.MoveToOuterDisposeScope();
			return output;
		}

		//internal MaskData _process_batch(BatchedInput batched, int[] crop_box, long orig_w, long orig_h)
		//{
		//	using var _ = no_grad();
		//	using var __ = NewDisposeScope();


		//	Tensor points = batched.Point_coords;
		//	batched.Point_coords = batched.Point_coords[.., TensorIndex.None, ..];
		//	batched.Point_labels = batched.Point_labels[.., TensorIndex.None];

		//	BatchedOutput output = model.forward(batched, true, true);
		//	output.to(CPU);

		//	Tensor indexs = torch.zeros(0);
		//	Tensor masks = output.Masks.flatten(0, 1);
		//	Tensor iouPredictions = output.Iou_predictions.flatten(0, 1);
		//	Tensor pts = points.repeat(new long[] { output.Masks.shape[1], 1 }).to(output.Iou_predictions.dtype, output.Iou_predictions.device);
		//	if (this.pred_iou_thresh > 0)
		//	{
		//		indexs = iouPredictions > this.pred_iou_thresh;
		//		masks = masks[indexs];
		//		iouPredictions = iouPredictions[indexs];
		//		pts = pts[indexs];

		//	}

		//	Tensor stability_score = Amg.calculate_stability_score(masks, model.mask_threshold, this.stability_score_offset);
		//	if (this.stability_score_thresh > 0)
		//	{
		//		indexs = stability_score >= stability_score_thresh;
		//		masks = masks[indexs];
		//		iouPredictions = iouPredictions[indexs];
		//		pts = pts[indexs];
		//		stability_score = stability_score[indexs];
		//	}

		//	masks = masks > model.mask_threshold;
		//	Tensor boxes = torchvision.ops.masks_to_boxes(masks);
		//	crop_box = crop_box ?? new int[] { 0, 0, (int)orig_w, (int)orig_h };
		//	indexs = ~Amg.is_box_near_crop_edge(boxes, crop_box, new int[] { 0, 0, (int)orig_w, (int)orig_h });
		//	if (!torch.all(indexs).ToBoolean())
		//	{
		//		indexs = stability_score >= stability_score_thresh;
		//		masks = masks[indexs];
		//		iouPredictions = iouPredictions[indexs];
		//		pts = pts[indexs];
		//		boxes = boxes[indexs];
		//		stability_score = stability_score[indexs];
		//	}

		//	// Compress to RLE
		//	masks = Amg.uncrop_masks(masks, crop_box, (int)orig_h, (int)orig_w);
		//	List<Rle> rles = Amg.mask_to_rle_pytorch(masks);
		//	masks.Dispose();
		//	output.Dispose();
		//	GC.Collect();

		//	MaskData mskData = new MaskData
		//	{
		//		Boxes = boxes.cpu().MoveToOuterDisposeScope(),
		//		IouPreds = iouPredictions.cpu().MoveToOuterDisposeScope(),
		//		Points = pts.cpu().MoveToOuterDisposeScope(),
		//		Rles = rles,
		//		StabilityScore = stability_score.cpu().MoveToOuterDisposeScope(),
		//	};
		//	return mskData;
		//}


		internal MaskData _process_batch(BatchedInput batched, int[] crop_box, long orig_w, long orig_h)
		{
			using var _ = no_grad();
			using var __ = NewDisposeScope();


			Tensor points = batched.Point_coords;
			batched.Point_coords = batched.Point_coords[.., TensorIndex.None, ..];
			batched.Point_labels = batched.Point_labels[.., TensorIndex.None];

			BatchedOutput output = model.forward(batched, true, true);
			output.to(CPU);

			Tensor indexs = torch.zeros(0);
			Tensor masks = output.Masks.flatten(0, 1);
			Tensor iouPredictions = output.Iou_predictions.flatten(0, 1);
			Tensor pts = points.repeat(new long[] { output.Masks.shape[1], 1 }).to(output.Iou_predictions.dtype, output.Iou_predictions.device);
			if (this.pred_iou_thresh > 0)
			{
				indexs = iouPredictions > this.pred_iou_thresh;
				masks = masks[indexs];
				iouPredictions = iouPredictions[indexs];
				pts = pts[indexs];
			}

			Tensor stability_score = Amg.calculate_stability_score(masks, model.mask_threshold, this.stability_score_offset);
			if (this.stability_score_thresh > 0)
			{
				indexs = stability_score >= stability_score_thresh;
				masks = masks[indexs];
				iouPredictions = iouPredictions[indexs];
				pts = pts[indexs];
				stability_score = stability_score[indexs];
			}

			masks = masks > model.mask_threshold;
			Tensor boxes = torchvision.ops.masks_to_boxes(masks);
			crop_box = crop_box ?? new int[] { 0, 0, (int)orig_w, (int)orig_h };
			indexs = ~Amg.is_box_near_crop_edge(boxes, crop_box, new int[] { 0, 0, (int)orig_w, (int)orig_h });
			if (!torch.all(indexs).ToBoolean())
			{
				indexs = stability_score >= stability_score_thresh;
				masks = masks[indexs];
				iouPredictions = iouPredictions[indexs];
				pts = pts[indexs];
				boxes = boxes[indexs];
				stability_score = stability_score[indexs];
			}

			output.Dispose();
			GC.Collect();

			return new MaskData
			{
				Boxes = boxes.cpu().MoveToOuterDisposeScope(),
				IouPreds = iouPredictions.cpu().MoveToOuterDisposeScope(),
				Points = pts.cpu().MoveToOuterDisposeScope(),
				Masks = masks.cpu().MoveToOuterDisposeScope(),
				StabilityScore = stability_score.cpu().MoveToOuterDisposeScope(),
			};
		}

		private void _process_crop(int[] crop_box, int layer)
		{

		}

	}
}
