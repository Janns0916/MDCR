import src.diffusion_model.gaussian_diffusion as gd
from src.diffusion_model.Denoise import DNN
import enum

# class ModelMeanType(enum.Enum):
#     START_X = enum.auto()  # the model predicts x_0
#     EPSILON = enum.auto()  # the model predicts epsilon


def denoise_out(args, device):
    # if args.mean_type == 'x0':
    #     mean_type = gd.ModelMeanType.START_X
    # elif args.mean_type == 'eps':
    #     mean_type = gd.ModelMeanType.EPSILON
    # else:
    #     raise ValueError("Unimplemented mean type %s" % args.mean_type)
    # out_dims = eval(args.dims) + [args.entity_num]
    out_dims = eval(args.dims) + [args.entity_num]
    in_dims = out_dims[::-1]
    # diffusion = gd.GaussianDiffusion(mean_type, args.noise_schedule, args.noise_scale, args.noise_min, args.noise_max,
    #                                  args.steps, device).to(device)
    diffusion = gd.GaussianDiffusion(gd.ModelMeanType.EPSILON, args.noise_schedule, args.noise_scale, args.noise_min, args.noise_max,
                                     args.steps, device).to(device)
    denoise = DNN(in_dims, out_dims, args.emb_size, time_type='cat', norm=args.norm).to(device)
    return diffusion, denoise