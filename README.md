# SamSharp

Run SAM(Segment-Anything) in C# with TorchSharp. </br>
With the help of this project you won't have to transform .pth model to onnx.

## Feature

- Written in C# only.
- Support Vit-b, Vit-l and Vit-h now.
- Support Load PreTrained models from SAM.
- Support .Net6 or higher.

## Models

You can download pre-trained models here.


| model | Download Link 
| --- | ----------- 
| vit-h | [ViT-H SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) 
| vit-l | [ViT-L SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth) 
| vit-b | [ViT-B SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) 



## How to use

You can download the code or add it from nuget.

    dotnet add package IntptrMax.SamSharp

> [!NOTE]
> Please add one of libtorch-cpu, libtorch-cuda-12.1, libtorch-cuda-12.1-win-x64 or libtorch-cuda-12.1-linux-x64 version 2.5.1.0 to execute.

In your code you can use it as below.

### Predict

You can use it with the code below:

```CSharp
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
```
And there is also a WinForm Demo.

![image](https://raw.githubusercontent.com/IntptrMax/SamSharp/refs/heads/master/Assets/Demo.jpg)
