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

<details>
  <summary>Checkpoints</summary>

| model | Download Link |
| --- | ----------- |
| vit-h | [ViT-H SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) |
| vit-l | [ViT-L SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth) |
| vit-b | [ViT-B SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) |

</details>


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
SamSharp.utils.SamPredictor predictor = new SamSharp.utils.SamPredictor(checkpointPath, device);

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
```
And there is also a WinForm Demo.

![image](https://raw.githubusercontent.com/IntptrMax/SamSharp/refs/heads/master/Assets/Demo.jpg)
