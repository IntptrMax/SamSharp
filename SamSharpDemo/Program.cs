using SkiaSharp;
using static SamSharp.Utils.Classes;

namespace SamSharpDemo
{
	internal class Program
	{
		static void Main(string[] args)
		{
			string checkpointPath = @".\Assets\sam_vit_b_01ec64.pth";
			SamDevice device = SamDevice.Cuda; // or SamDevice.Cpu if you want to run on CPU
			SamSharp.Utils.SamPredictor predictor = new SamSharp.Utils.SamPredictor(checkpointPath, device);

			SKBitmap image = SKBitmap.Decode(@".\Assets\demo.jpg");
			List<SamPoint> points = new List<SamPoint>
			{
				new SamPoint(500, 375, true), // Foreground point
			};
			(List<SKBitmap> bitmaps, List<float[]> iou_predictions) =  predictor.Predict(image, points);
			Console.WriteLine("The predictions are :");
			foreach (float pred in iou_predictions[0])
			{
				Console.WriteLine((pred * 100).ToString("f2") + "%");
			}
			SKBitmap resultImg = bitmaps[0];
			using (var skData = resultImg.Encode(SKEncodedImageFormat.Jpeg, 80))
			{
				using (var stream = File.OpenWrite(@"mask.jpg"))
				{
					skData.SaveTo(stream);
				}
			}

		}
	}
}
