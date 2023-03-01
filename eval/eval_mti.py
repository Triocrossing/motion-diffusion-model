from utils.parser_util import evaluation_parser
from utils.fixseed import fixseed
from datetime import datetime
from data_loaders.humanml.motion_loaders.model_motion_loaders import (
    get_mdm_loader,
    get_mti_loader,
)  # get_motion_loader
from data_loaders.humanml.utils.metrics import *
from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
from collections import OrderedDict
from data_loaders.humanml.scripts.motion_process import *
from data_loaders.humanml.utils.utils import *
from utils.model_util import create_model_and_diffusion, load_model_wo_clip

from diffusion import logger
from utils import dist_util
from data_loaders.get_data import get_dataset_loader
from model.cfg_sampler import ClassifierFreeSampleModel, NoClassifierFreeSampleModel

torch.multiprocessing.set_sharing_strategy("file_system")
import os.path as osp
import pickle

viz_motion = False


def load_pickle(pickle_path):
    """Load a pickle file."""
    if not osp.exists(pickle_path):
        return None
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    return data


def check_percentage(
    fname,
    folder_dir="/media/xi/ssd02/Work/textual_inversion/benchmark/save_motion",
):
    ctr = 0
    ctrs = []
    for idx, name in enumerate(fname):
        randidx = np.random.randint(10)
        _motion = load_pickle(osp.join(folder_dir, f"{name}_{randidx}.pkl"))

        if _motion is not None:
            ctr += 1
            ctrs.append(1)
        else:
            ctrs.append(0)
    print(f"percentage of motion existed: {float(ctr)/len(fname)}")
    return ctrs


def load_motion(
    fname,
    motions,
    mean_for_eval,
    std_for_eval,
    folder_dir="/media/xi/ssd02/Work/textual_inversion/benchmark/save_motion",
):
    ctr = 0
    ctrs = []

    for idx, name in enumerate(fname):
        randidx = np.random.randint(10)
        _motion = load_pickle(osp.join(folder_dir, f"{name}_{randidx}.pkl"))

        if _motion is not None:
            renormed_motion = (_motion - mean_for_eval) / std_for_eval
            motions[idx] = renormed_motion
            ctr += 1
            ctrs.append(1)
        else:
            ctrs.append(0)
    print(f"percentage of motion existed: {float(ctr)/len(fname)}")
    return motions, ctrs


import numpy as np


if viz_motion:

    from data_loaders.humanml.utils.plot_script import plot_3d_motion
    import data_loaders.humanml.utils.paramUtil as paramUtil
    from data_loaders.humanml.scripts.motion_process import recover_from_ric
    from model.rotation2xyz import Rotation2xyz

    skeleton = paramUtil.t2m_kinematic_chain
    _out_path = "/media/xi/ssd02/Work/textual_inversion/benchmark/viz"
    n_joints = 22  # if sample.shape[1] == 263 else 21

    def vis_generated(
        sample,
        out_path,
        motion_loader,
        motion_loader_name,
        text,
        fname,
        ctrs,
        model_kwargs=None,
        batch_size=1,
        n_frames=200,
        inbetween_mode=False,  # FIXME ..
    ):
        rot2xyz = Rotation2xyz(device="cpu", dataset="humanml")
        # for sample in samples:
        if motion_loader_name == "ground truth":
            print("vis gt")
            return
            sample = motion_loader.dataset.t2m_dataset.inv_transform(
                sample.cpu().permute(0, 2, 3, 1)
            ).float()
        else:
            print("vis gen")
            sample = (
                sample.cpu().permute(0, 2, 3, 1)
                * motion_loader.dataset.dataset.std_for_eval
                + motion_loader.dataset.dataset.mean_for_eval
            ).float()
            # sample = sample.cpu().permute(0, 2, 3, 1).float()
            # according to T2M norms
            # motion = renormed_motion
            # sample = motion_loader.dataset.dataset.t2m_dataset.inv_transform(
            # sample.cpu().permute(0, 2, 3, 1)
            # ).float()
        sample = recover_from_ric(sample, n_joints)  # torch.Size([10, 1, 60, 22, 3])
        # batch size, 1, frames,
        sample = sample.view(-1, *sample.shape[2:]).permute(
            0, 2, 3, 1
        )  # torch.Size([10, 22, 3, 60])
        rot2xyz_pose_rep = "xyz"  # (
        #     "xyz"
        #     if self.model.data_rep in ["xyz", "hml_vec"]
        #     else self.model.data_rep
        # )
        rot2xyz_mask = (
            None
            if rot2xyz_pose_rep == "xyz"
            else model_kwargs["y"]["mask"].reshape(batch_size, n_frames).bool()
        )
        sample = rot2xyz(
            x=sample,
            mask=rot2xyz_mask,
            pose_rep=rot2xyz_pose_rep,
            glob=True,
            translation=True,
            jointstype="smpl",
            vertstrans=True,
            betas=None,
            beta=0,
            glob_rot=None,
            get_rotations_back=False,
        )  # torch.Size([10, 22, 3, 60])

        # all_samples.append(sample)

        # for step, pred_xstart in enumerate(all_out):
        for idx, _sample in enumerate(sample):
            _sample = _sample.numpy().transpose(2, 0, 1)
            _text = text[idx]
            # _fname = fname[idx]
            caption = f"{fname[idx]}_{motion_loader_name}"
            _ext = "_a" if "*" in _text else ""
            _ext += "_inb_" if inbetween_mode else ""
            if ctrs[idx] == 1:
                _ext += "_m_saved"
            animation_save_path = osp.join(out_path, caption + _ext + ".mp4")
            if inbetween_mode:
                pref = 0.25
                suffix = 0.75
                gt_frames = list(range(n_frames))[
                    int(n_frames * pref) : int(n_frames * suffix)
                ]
            else:
                gt_frames = []
            plot_3d_motion(
                save_path=animation_save_path,
                kinematic_tree=skeleton,
                joints=_sample,
                dataset="humanml",  # self.dataset.name,
                title=_text,
                fps=15,  # self.dataset.fps,
                gt_frames=gt_frames,
            )

    # def viz_motion_mp4(motion, caption, dataset=None, out_path=_out_path):
    #     motion = motion.numpy().transpose(2, 0, 1)
    #     # print(motion)
    #     # caption = f"{model}_{id}"
    #     save_file = caption + ".mp4"
    #     animation_save_path = osp.join(out_path, save_file)
    #     plot_3d_motion(
    #         animation_save_path,
    #         skeleton,
    #         motion,
    #         dataset="humanml",  # dataset.name,
    #         title=caption,
    #         fps=15,  # dataset.fps,
    #     )


def evaluate_matching_score(eval_wrapper, motion_loaders, file):
    match_score_dict = OrderedDict({})
    R_precision_dict = OrderedDict({})
    activation_dict = OrderedDict({})
    # ad hoc
    gt_text_embedding = None
    gt_word_embeddings = None
    gt_pos_one_hots = None
    gt_sent_lens = None
    # gt_text_embedding = None
    print("========== Evaluating Matching Score ==========")

    for motion_loader_name, motion_loader in motion_loaders.items():
        all_motion_embeddings = []
        score_list = []
        all_size = 0
        matching_score_sum = 0
        top_k_count = 0
        ctrs = None
        # print(motion_loader_name)
        with torch.no_grad():
            for idx, batch in enumerate(motion_loader):
                (
                    word_embeddings,
                    # pseudo_word_embeddings,?
                    pos_one_hots,
                    text,
                    sent_lens,
                    motions,
                    m_lens,
                    *_,
                    fname,
                ) = batch

                # import ipdb

                # ipdb.set_trace()
                if motion_loader_name != "ground truth":
                    mean_for_eval = motion_loader.dataset.dataset.mean_for_eval
                    std_for_eval = motion_loader.dataset.dataset.std_for_eval

                    motions, ctrs = load_motion(
                        fname, motions, mean_for_eval, std_for_eval
                    )

                # if viz motion
                # global viz_motion
                print(fname)

                if viz_motion:
                    # for idx in range(motions.shape[0]):
                    # viz_motion = False  # gen once
                    # import ipdb

                    # ipdb.set_trace()
                    vis_generated(
                        sample=motions.unsqueeze(2).permute(0, 3, 2, 1),
                        out_path=_out_path,
                        motion_loader=motion_loader,
                        motion_loader_name=motion_loader_name,
                        text=text,
                        fname=fname,
                        ctrs=ctrs,
                    )

                # text_embeddings = a person <>
                # pseudo_text_embeddings = embedded GT text

                text_embeddings, motion_embeddings = eval_wrapper.get_co_embeddings(
                    word_embs=word_embeddings,
                    pos_ohot=pos_one_hots,
                    cap_lens=sent_lens,
                    motions=motions,
                    m_lens=m_lens,
                )
                dist_mat = euclidean_distance_matrix(
                    text_embeddings.cpu().numpy(), motion_embeddings.cpu().numpy()
                )
                matching_score_sum += dist_mat.trace()

                argsmax = np.argsort(dist_mat, axis=1)
                top_k_mat = calculate_top_k(argsmax, top_k=3)
                top_k_count += top_k_mat.sum(axis=0)

                all_size += text_embeddings.shape[0]

                all_motion_embeddings.append(motion_embeddings.cpu().numpy())

            all_motion_embeddings = np.concatenate(all_motion_embeddings, axis=0)
            matching_score = matching_score_sum / all_size
            R_precision = top_k_count / all_size
            match_score_dict[motion_loader_name] = matching_score
            R_precision_dict[motion_loader_name] = R_precision
            activation_dict[motion_loader_name] = all_motion_embeddings

        print(f"---> [{motion_loader_name}] Matching Score: {matching_score:.4f}")
        print(
            f"---> [{motion_loader_name}] Matching Score: {matching_score:.4f}",
            file=file,
            flush=True,
        )

        line = f"---> [{motion_loader_name}] R_precision: "
        for i in range(len(R_precision)):
            line += "(top %d): %.4f " % (i + 1, R_precision[i])
        print(line)
        print(line, file=file, flush=True)

    return match_score_dict, R_precision_dict, activation_dict


def evaluate_fid(eval_wrapper, groundtruth_loader, activation_dict, file):
    eval_dict = OrderedDict({})
    gt_motion_embeddings = []
    print("========== Evaluating FID ==========")
    with torch.no_grad():
        for idx, batch in enumerate(groundtruth_loader):
            _, _, _, sent_lens, motions, m_lens, _, *_ = batch
            motion_embeddings = eval_wrapper.get_motion_embeddings(
                motions=motions, m_lens=m_lens
            )
            gt_motion_embeddings.append(motion_embeddings.cpu().numpy())
    gt_motion_embeddings = np.concatenate(gt_motion_embeddings, axis=0)
    gt_mu, gt_cov = calculate_activation_statistics(gt_motion_embeddings)

    # print(gt_mu)
    for model_name, motion_embeddings in activation_dict.items():
        mu, cov = calculate_activation_statistics(motion_embeddings)
        # print(mu)
        fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
        print(f"---> [{model_name}] FID: {fid:.4f}")
        print(f"---> [{model_name}] FID: {fid:.4f}", file=file, flush=True)
        eval_dict[model_name] = fid
    return eval_dict


def evaluate_diversity(activation_dict, file, diversity_times):
    eval_dict = OrderedDict({})
    print("========== Evaluating Diversity ==========")
    for model_name, motion_embeddings in activation_dict.items():
        diversity = calculate_diversity(motion_embeddings, diversity_times)
        eval_dict[model_name] = diversity
        print(f"---> [{model_name}] Diversity: {diversity:.4f}")
        print(f"---> [{model_name}] Diversity: {diversity:.4f}", file=file, flush=True)
    return eval_dict


def evaluate_multimodality(eval_wrapper, mm_motion_loaders, file, mm_num_times):
    eval_dict = OrderedDict({})
    print("========== Evaluating MultiModality ==========")
    for model_name, mm_motion_loader in mm_motion_loaders.items():
        mm_motion_embeddings = []
        with torch.no_grad():
            for idx, batch in enumerate(mm_motion_loader):
                # (1, mm_replications, dim_pos)
                motions, m_lens = batch
                motion_embedings = eval_wrapper.get_motion_embeddings(
                    motions[0], m_lens[0]
                )
                mm_motion_embeddings.append(motion_embedings.unsqueeze(0))
        if len(mm_motion_embeddings) == 0:
            multimodality = 0
        else:
            mm_motion_embeddings = torch.cat(mm_motion_embeddings, dim=0).cpu().numpy()
            multimodality = calculate_multimodality(mm_motion_embeddings, mm_num_times)
        print(f"---> [{model_name}] Multimodality: {multimodality:.4f}")
        print(
            f"---> [{model_name}] Multimodality: {multimodality:.4f}",
            file=file,
            flush=True,
        )
        eval_dict[model_name] = multimodality
    return eval_dict


def get_metric_statistics(values, replication_times):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    conf_interval = 1.96 * std / np.sqrt(replication_times)
    return mean, conf_interval


# TODO: viz motion
# def viz over the motion_loader


def evaluation(
    eval_wrapper,
    gt_loader,
    eval_motion_loaders,
    log_file,
    replication_times,
    diversity_times,
    mm_num_times,
    run_mm=False,
):
    with open(log_file, "w") as f:
        all_metrics = OrderedDict(
            {
                "Matching Score": OrderedDict({}),
                "R_precision": OrderedDict({}),
                "FID": OrderedDict({}),
                "Diversity": OrderedDict({}),
                "MultiModality": OrderedDict({}),
            }
        )
        for replication in range(replication_times):
            motion_loaders = {}
            # multimodality
            mm_motion_loaders = {}
            motion_loaders["ground truth"] = gt_loader
            for motion_loader_name, motion_loader_getter in eval_motion_loaders.items():
                motion_loader, mm_motion_loader = motion_loader_getter()
                motion_loaders[motion_loader_name] = motion_loader
                mm_motion_loaders[motion_loader_name] = mm_motion_loader

            print(
                f"==================== Replication {replication} ===================="
            )
            print(
                f"==================== Replication {replication} ====================",
                file=f,
                flush=True,
            )
            print(f"Time: {datetime.now()}")
            print(f"Time: {datetime.now()}", file=f, flush=True)
            mat_score_dict, R_precision_dict, acti_dict = evaluate_matching_score(
                eval_wrapper, motion_loaders, f
            )

            print(f"Time: {datetime.now()}")
            print(f"Time: {datetime.now()}", file=f, flush=True)
            fid_score_dict = evaluate_fid(eval_wrapper, gt_loader, acti_dict, f)

            print(f"Time: {datetime.now()}")
            print(f"Time: {datetime.now()}", file=f, flush=True)
            div_score_dict = evaluate_diversity(acti_dict, f, diversity_times)

            if run_mm:
                print(f"Time: {datetime.now()}")
                print(f"Time: {datetime.now()}", file=f, flush=True)
                mm_score_dict = evaluate_multimodality(
                    eval_wrapper, mm_motion_loaders, f, mm_num_times
                )

            print(f"!!! DONE !!!")
            print(f"!!! DONE !!!", file=f, flush=True)

            for key, item in mat_score_dict.items():
                if key not in all_metrics["Matching Score"]:
                    all_metrics["Matching Score"][key] = [item]
                else:
                    all_metrics["Matching Score"][key] += [item]

            for key, item in R_precision_dict.items():
                if key not in all_metrics["R_precision"]:
                    all_metrics["R_precision"][key] = [item]
                else:
                    all_metrics["R_precision"][key] += [item]

            for key, item in fid_score_dict.items():
                if key not in all_metrics["FID"]:
                    all_metrics["FID"][key] = [item]
                else:
                    all_metrics["FID"][key] += [item]

            for key, item in div_score_dict.items():
                if key not in all_metrics["Diversity"]:
                    all_metrics["Diversity"][key] = [item]
                else:
                    all_metrics["Diversity"][key] += [item]
            if run_mm:
                for key, item in mm_score_dict.items():
                    if key not in all_metrics["MultiModality"]:
                        all_metrics["MultiModality"][key] = [item]
                    else:
                        all_metrics["MultiModality"][key] += [item]

        # print(all_metrics['Diversity'])
        mean_dict = {}
        for metric_name, metric_dict in all_metrics.items():
            print("========== %s Summary ==========" % metric_name)
            print("========== %s Summary ==========" % metric_name, file=f, flush=True)
            for model_name, values in metric_dict.items():
                # print(metric_name, model_name)
                mean, conf_interval = get_metric_statistics(
                    np.array(values), replication_times
                )
                mean_dict[metric_name + "_" + model_name] = mean
                # print(mean, mean.dtype)
                if isinstance(mean, np.float64) or isinstance(mean, np.float32):
                    print(
                        f"---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}"
                    )
                    print(
                        f"---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}",
                        file=f,
                        flush=True,
                    )
                elif isinstance(mean, np.ndarray):
                    line = f"---> [{model_name}]"
                    for i in range(len(mean)):
                        line += "(top %d) Mean: %.4f CInt: %.4f;" % (
                            i + 1,
                            mean[i],
                            conf_interval[i],
                        )
                    print(line)
                    print(line, file=f, flush=True)
        return mean_dict


if __name__ == "__main__":
    args = evaluation_parser()
    fixseed(args.seed)
    args.batch_size = 32  # This must be 32! Don't change it! otherwise it will cause a bug in R precision calc!
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace("model", "").replace(".pt", "")
    log_file = os.path.join(
        os.path.dirname(args.model_path), "eval_humanml_{}_{}".format(name, niter)
    )
    if args.guidance_param != 1.0:
        log_file += f"_gscale{args.guidance_param}"
    log_file += f"_{args.eval_mode}"
    log_file += ".log"

    print(f"Will save to log file [{log_file}]")

    print(f"Eval mode [{args.eval_mode}]")
    if args.eval_mode == "debug":
        num_samples_limit = 1000  # None means no limit (eval over all dataset)
        run_mm = False
        mm_num_samples = 0
        mm_num_repeats = 0
        mm_num_times = 0
        diversity_times = 300
        replication_times = 5  # about 3 Hrs
    elif args.eval_mode == "wo_diversity":
        num_samples_limit = 1000  # None means no limit (eval over all dataset)
        run_mm = False
        mm_num_samples = 0
        mm_num_repeats = 0
        mm_num_times = 0
        diversity_times = 30
        replication_times = 5  #
    elif args.eval_mode == "wo_mm":
        num_samples_limit = 1000
        run_mm = False
        mm_num_samples = 0
        mm_num_repeats = 0
        mm_num_times = 0
        diversity_times = 300
        replication_times = 20  # about 12 Hrs
    elif args.eval_mode == "mm_short":
        num_samples_limit = 1000
        run_mm = True
        mm_num_samples = 100
        mm_num_repeats = 30
        mm_num_times = 10
        diversity_times = 300
        replication_times = 5  # about 15 Hrs
    else:
        raise ValueError()

    dist_util.setup_dist(args.device)
    logger.configure()

    logger.log("creating data loader...")
    split = "test"

    # gen_loader = get_dataset_loader(
    #     name=args.dataset,
    #     batch_size=args.batch_size,
    #     num_frames=None,
    #     split=split,
    #     hml_mode="eval",
    # )
    # cgen_loader = get_dataset_loader(
    #     name=args.dataset,
    #     batch_size=args.batch_size,
    #     num_frames=None,
    #     split=split,
    #     # hml_mode="eval_mti",
    #     hml_mode="eval",
    # )

    gen_hml_mode = "eval_mti" if args.mti else "eval"
    # print("eval mode, (not mti)")
    cgen_loader = get_dataset_loader(
        name=args.dataset,
        batch_size=args.batch_size,
        num_frames=None,
        split=split,
        hml_mode=gen_hml_mode,
        # hml_mode="eval",
    )

    gt_loader = get_dataset_loader(
        name=args.dataset,
        batch_size=args.batch_size,
        num_frames=None,
        split=split,
        hml_mode="gt",
    )

    num_actions = cgen_loader.dataset.num_actions

    logger.log("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, cgen_loader)

    logger.log(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location="cpu")
    load_model_wo_clip(model, state_dict)

    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(
            model
        )  # wrapping model with the classifier-free sampler
    else:
        model = NoClassifierFreeSampleModel(model)
    model.to(dist_util.dev())
    model.eval()  # disable random masking

    lder_name = gen_hml_mode
    lder_name += "_inb" if args.inbetween_mode else ""
    eval_motion_loaders = {
        ################
        ## HumanML3D Dataset##
        ################
        lder_name: lambda: get_mti_loader(
            model,
            diffusion,
            args.batch_size,
            cgen_loader,
            mm_num_samples,
            mm_num_repeats,
            gt_loader.dataset.opt.max_motion_length,
            num_samples_limit,
            args.guidance_param,
            args.inbetween_mode,
        )
    }

    eval_wrapper = EvaluatorMDMWrapper(args.dataset, dist_util.dev())

    evaluation(
        eval_wrapper,
        gt_loader,
        eval_motion_loaders,
        log_file,
        replication_times,
        diversity_times,
        mm_num_times,
        run_mm=run_mm,
    )