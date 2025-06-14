using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace SamSharp.Modeling
{
	internal class Common
	{
		internal class MLPBlock : Module<Tensor, Tensor>
		{
			public enum ActivationType
			{
				GELU,
				ReLU,
				SiLU,
			}

			private readonly Linear lin1;
			private readonly Linear lin2;
			private readonly Module<Tensor,Tensor> act;

			public MLPBlock(int embedding_dim, int mlp_dim, ActivationType activationType = ActivationType.GELU) : base(nameof(MLPBlock))
			{
				this.lin1 = nn.Linear(embedding_dim, mlp_dim);
				this.lin2 = nn.Linear(mlp_dim, embedding_dim);
				this.act =  activationType switch
				{
					ActivationType.GELU => GELU(),
					ActivationType.ReLU => ReLU(),
					ActivationType.SiLU => SiLU(),
					_ => throw new ArgumentException("Unsupported activation type", nameof(activationType)),
				};
				RegisterComponents();
			}

			public override Tensor forward(Tensor x)
			{
				return this.lin2.forward(this.act.forward(this.lin1.forward(x)));
			}
		}

		internal class LayerNorm2d : Module<Tensor, Tensor>
		{
			private readonly Parameter weight;
			private readonly Parameter bias;
			private readonly float eps;

			public LayerNorm2d(int num_channels, float eps = 1e-6f) : base(nameof(LayerNorm2d))
			{
				this.weight = nn.Parameter(torch.ones(num_channels));
				this.bias = nn.Parameter(torch.zeros(num_channels));
				this.eps = eps;
				RegisterComponents();
			}

			public override Tensor forward(Tensor x)
			{
				using var _ = NewDisposeScope();
				Tensor u = x.mean(new long[] { 1 }, keepdim: true);
				Tensor s = (x - u).pow(2).mean(new long[] { 1 }, keepdim: true);
				x = (x - u) / torch.sqrt(s + this.eps);
				x = this.weight[.., TensorIndex.Null, TensorIndex.Null] * x + this.bias[.., TensorIndex.Null, TensorIndex.Null];
				return x.MoveToOuterDisposeScope();
			}

		}
	}
}
