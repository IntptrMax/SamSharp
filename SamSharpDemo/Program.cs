using SkiaSharp;
using System.Threading.Tasks;
using static SamSharp.Utils.Classes;

namespace SamSharpDemo
{
	internal class Program
	{
		static void Main(string[] args)
		{
			string checkpointPath = @".\Assets\sam_vit_b_01ec64.pth";
			SamDevice device = SamDevice.Cuda; // or SamDevice.Cpu if you want to run on CPU
			int imageSize = 512; // The maximum size of the image to process, can be adjusted based on your GPU memory
			SamSharp.Utils.SamPredictor predictor = new SamSharp.Utils.SamPredictor(checkpointPath, device);

			SKBitmap image = SKBitmap.Decode(@".\Assets\truck.jpg");

			List<SamPoint> points = new List<SamPoint>
			{
				new SamPoint(500, 375, false),
				new SamPoint(1524,675, false),
			};
			List<SamBox> boxes = new List<SamBox>
			{
				new SamBox(75, 275, 1725, 850),
				new SamBox(425, 600, 700, 875),
				new SamBox(1375, 550, 1650, 800),
				new SamBox(1240, 675, 1400, 750),
			};

			List<PredictOutput> outputs = predictor.Predict(image, points, boxes, imageSize);
			Console.WriteLine("The predictions are :");

			using (SKCanvas canvas = new SKCanvas(image))
			{
				canvas.Clear(SKColors.Transparent);
				for (int i = 0; i < outputs.Count; i++)
				{
					PredictOutput output = outputs[i];
					Console.WriteLine($"Mask {i}: Precision: {output.Precision * 100:F2}%");
					bool[,] mask = output.Mask;
					Random random = new Random();
					SKColor color = new SKColor((byte)random.Next(256), (byte)random.Next(256), (byte)random.Next(256));
					for (int y = 0; y < mask.GetLength(1); y++)
					{
						for (int x = 0; x < mask.GetLength(0); x++)
						{
							if (mask[x, y])
							{
								canvas.DrawPoint(x, y, color);
							}
						}
					}
				}
			}
			SKData data = image.Encode(SKEncodedImageFormat.Png, 100);
			using (var stream = File.OpenWrite($"mask.png"))
			{
				data.SaveTo(stream);
			}

		}
	}
}
