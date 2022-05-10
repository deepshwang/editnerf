from .models.base_model import NeRF
from .models.embedders import get_embedder
import torch
import os

SEED = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn

    def ret(inputs, styles, alpha, feature, maskMLP=None, mask=None, mapperMLP=None):
        results = []
        masks = []
        for i in range(0, inputs.shape[0], chunk):
            input_chunk = inputs[i:i + chunk]
            style_chunk = styles[i:i + chunk]
            alpha_chunk = alpha[i:i + chunk] if alpha is not None else None
            feature_chunk = feature[i:i + chunk] if feature is not None else None
            mask = mask[i:i + chunk] if mask is not None else None
            if maskMLP is None:
                results.append(fn(input_chunk, style_chunk, alpha=alpha_chunk, feature=feature_chunk, maskMLP=maskMLP, mask=mask, mapperMLP=mapperMLP))
            else:
                result, mask = fn(input_chunk, style_chunk, alpha=alpha_chunk, feature=feature_chunk, maskMLP=maskMLP)
                results.append(result)
                masks.append(mask)
                #return torch.cat(results, 0)
        if maskMLP is None:
            return torch.cat(results, 0)
        else:
            return torch.cat(results, 0), torch.cat(masks, 0)
    return ret


def run_network(inputs, styles, viewdirs, fn, alpha, feature, embed_fn, embeddirs_fn, netchunk=1024 * 64, maskMLP=None, mask=None, mapperMLP=None):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)
    if viewdirs is not None:
        input_dirs = viewdirs[:, None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)
    if alpha is not None:
        alpha = torch.reshape(alpha, [-1, 1])
    if feature is not None:
        feature = torch.reshape(feature, [-1, feature.shape[-1]])
    if maskMLP is None:
        if mask is not None: 
            mask = torch.unsqueeze(torch.flatten(mask), -1)
        outputs_flat = batchify(fn, netchunk)(embedded, styles, alpha, feature, maskMLP=maskMLP, mask=mask, mapperMLP=mapperMLP)
        outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
        return outputs
    else:
        outputs_flat, mask_outputs_flat = batchify(fn, netchunk)(embedded, styles, alpha, feature, maskMLP=maskMLP)
        outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
        mask_outputs = torch.reshape(mask_outputs_flat, list(inputs.shape[:-1]) + [mask_outputs_flat.shape[-1]])
        return outputs, torch.squeeze(mask_outputs)

def load_checkpoint(chkpt_dir, args):
    ckpts = [os.path.join(chkpt_dir, f) for f in sorted(os.listdir(chkpt_dir)) if 'tar' in f]

    if not args.no_reload and (args.load_it != 0 or len(ckpts) > 0):
        if args.load_it != 0:
            ckpt_path = os.path.join(chkpt_dir, '{:06d}.tar'.format(args.load_it))
        else:
            ckpt_path = ckpts[-1]

        ckpt = torch.load(ckpt_path)
        return ckpt
    else:
        return None


def create_nerf(args, return_styles=False):
    """Instantiate NeRF's MLP model.
    """
    if SEED:
        torch.manual_seed(1234)

    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    style_dim = args.style_dim

    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 5 if args.N_importance > 0 else 4
    model = NeRF(D_mean=args.D_mean, W_mean=args.W_mean, D_instance=args.D_instance, W_instance=args.W_instance, D_fusion=args.D_fusion, W_fusion=args.W_fusion, D_sigma=args.D_sigma,
                 D_rgb=args.D_rgb, W_rgb=args.W_rgb, W_bottleneck=args.W_bottleneck, input_ch=input_ch, output_ch=output_ch, input_ch_views=input_ch_views, style_dim=style_dim,
                 embed_dim=args.embed_dim, style_depth=args.style_depth, shared_shape=args.shared_shape, use_viewdirs=args.use_viewdirs, separate_codes=args.separate_codes, use_styles=args.use_styles).to(device)

    grad_vars = list(model.parameters())

    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(D_mean=args.D_mean, W_mean=args.W_mean, D_instance=args.D_instance, W_instance=args.W_instance, D_fusion=args.D_fusion, W_fusion=args.W_fusion, D_sigma=args.D_sigma,
                          D_rgb=args.D_rgb, W_rgb=args.W_rgb, W_bottleneck=args.W_bottleneck, input_ch=input_ch, output_ch=output_ch, input_ch_views=input_ch_views, style_dim=style_dim, embed_dim=args.embed_dim, style_depth=args.style_depth, shared_shape=args.shared_shape, use_viewdirs=args.use_viewdirs, separate_codes=args.separate_codes, use_styles=args.use_styles).to(device)
        grad_vars += list(model_fine.parameters())

    def network_query_fn(inputs, styles, viewdirs, network_fn, alpha, feature, maskMLP=None, mask=None, mapperMLP=None): return run_network(inputs, styles, viewdirs, network_fn, alpha, feature,
                                                                                                   embed_fn=embed_fn,
                                                                                                   embeddirs_fn=embeddirs_fn,
                                                                                                   netchunk=args.netchunk,
                                                                                                   maskMLP=maskMLP,
                                                                                                   mask=mask,
                                                                                                   mapperMLP=mapperMLP)

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    ckpt = load_checkpoint(os.path.join(basedir, expname), args)  # Load checkpoints
    if args.load_from is not None:
        print('Loading from', args.load_from)
        ckpt = torch.load(args.load_from)

    if ckpt is not None and not args.skip_loading:
        start = ckpt['global_step']
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    ##########################

    render_kwargs_train = {
        'network_query_fn': network_query_fn,
        'perturb': args.perturb,
        'perturb_coarse': args.perturb_coarse,
        'N_importance': args.N_importance,
        'network_fine': model_fine,
        'N_samples': args.N_samples,
        'network_fn': model,
        'use_viewdirs': args.use_viewdirs,
        'white_bkgd': args.white_bkgd,
        'raw_noise_std': args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    render_kwargs_train['ndc'] = False
    render_kwargs_train['lindisp'] = False

    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['perturb_coarse'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    if return_styles:
        return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, ckpt['styles'][:args.N_instances]

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer
