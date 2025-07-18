# SamSharp

Run SAM(Segment-Anything) in C# with TorchSharp. </br>
With the help of this project you won't have to transform .pth model to onnx.

## Feature

- Written in C# only.
- Support Vit-b, Vit-l and Vit-h and MobileSam now.
- Support Load PreTrained models from SAM.
- Support .Net6 or higher.
- Support CPU and CUDA.
- Support Float32 and Float16 data type.
- Support Automatic Mask Generator.
- Support Predict with points and boxes.

## Models

You can download pre-trained models here.


| model | Download Link 
| --- | ----------- 
| vit-h | [ViT-H SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) 
| vit-l | [ViT-L SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth) 
| vit-b | [ViT-B SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) 
| vit-t | [ViT-T Mobile Sam model](https://huggingface.co/dhkim2810/MobileSAM/resolve/main/mobile_sam.pt) 



## How to use

You can download the code or add it from nuget.

    dotnet add package IntptrMax.SamSharp

> [!NOTE]
> Please add one of libtorch-cpu, libtorch-cuda-12.1, libtorch-cuda-12.1-win-x64 or libtorch-cuda-12.1-linux-x64 version 2.5.1.0 to execute.

In your code you can use it as below.

### Predict

```CSharp
string checkpointPath = @"..\..\..\Assets\Weights\MobileSam.pt";
SamDevice device = SamDevice.CUDA; // or SamDevice.Cpu if you want to run on CPU
SamScalarType dtype = SamScalarType.Float32;
int imageSize = 512; // The maximum size of the image to process, can be adjusted based on your GPU memory

SKBitmap image = SKBitmap.Decode(@"..\..\..\Assets\Images\truck.jpg");

// Use predictor
SamSharp.Utils.SamPredictor predictor = new SamSharp.Utils.SamPredictor(checkpointPath, device, dtype);
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

predictor.SetImage(image);
List<PredictOutput> outputs = predictor.Predict(null, boxes);

Console.WriteLine("The predictions are :");

using (SKCanvas canvas = new SKCanvas(image))
{
	canvas.Clear(SKColors.Transparent);
	var random = new Random();

	for (int i = 0; i < outputs.Count; i++)
	{
		PredictOutput output = outputs[i];
		Console.WriteLine($"Mask {i}: Precision: {output.Precision * 100:F2}%");
		bool[,] mask = output.Mask;

		SKColor color = new SKColor((byte)random.Next(256), (byte)random.Next(256), (byte)random.Next(256));
		using (var paint = new SKPaint { Color = color, BlendMode = SKBlendMode.Src })
		{
			int width = mask.GetLength(0);
			int height = mask.GetLength(1);

			using (var path = new SKPath())
			{
				for (int y = 0; y < height; y++)
				{
					for (int x = 0; x < width; x++)
					{
						if (mask[x, y])
						{
							path.AddRect(new SKRect(x, y, x + 1, y + 1));
						}
					}
				}
				canvas.DrawPath(path, paint);
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

### Automatic Mask Generator
```CSharp
string checkpointPath = @".\Assets\sam_vit_h_4b8939.pth";
SamDevice device = SamDevice.CUDA; // or SamDevice.Cpu if you want to run on CPU
SamScalarType dtype = SamScalarType.Float32;
int imageSize = 512; // The maximum size of the image to process, can be adjusted based on your GPU memory

SKBitmap image = SKBitmap.Decode(@"..\..\..\Assets\dog.jpg");

// Use Automatic Mask Generator
SamSharp.Utils.SamAutomaticMaskGenerator generator = new SamSharp.Utils.SamAutomaticMaskGenerator(checkpointPath, device: device, dtype: dtype);
List<PredictOutput> outputs = generator.generate(image, maxImageSize: imageSize);

Console.WriteLine("The predictions are :");

using (SKCanvas canvas = new SKCanvas(image))
{
	canvas.Clear(SKColors.Transparent);
	var random = new Random();

	for (int i = 0; i < outputs.Count; i++)
	{
		PredictOutput output = outputs[i];
		Console.WriteLine($"Mask {i}: Precision: {output.Precision * 100:F2}%");
		bool[,] mask = output.Mask;

		SKColor color = new SKColor((byte)random.Next(256), (byte)random.Next(256), (byte)random.Next(256));
		using (var paint = new SKPaint { Color = color, BlendMode = SKBlendMode.Src })
		{
			int width = mask.GetLength(0);
			int height = mask.GetLength(1);

			using (var path = new SKPath())
			{
				for (int y = 0; y < height; y++)
				{
					for (int x = 0; x < width; x++)
					{
						if (mask[x, y])
						{
							path.AddRect(new SKRect(x, y, x + 1, y + 1));
						}
					}
				}
				canvas.DrawPath(path, paint);
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
The result mask is.
![image](https://raw.githubusercontent.com/IntptrMax/SamSharp/refs/heads/master/Assets/mask_dog.png)


## Work to do
- [ ] Speed up.
- [ ] Use less VRAM.
- [ ] Use less RAM.